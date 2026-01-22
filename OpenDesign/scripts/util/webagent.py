import os
import re
import time
import uuid
import shutil
import random
import logging
import threading
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    WebDriverException,
    TimeoutException,
    NoSuchElementException,
    ElementNotInteractableException,
    UnexpectedAlertPresentException,
    NoAlertPresentException,
)

from util.azure_gpt4o import Openai as AzureOpenai
from util.azure_gpt4o import _get_api_infos_from_config

# Get API configuration from config.yaml for interactive judge
API_INFOS = _get_api_infos_from_config('interactive')

from util.arena_judge_prompt import SYSTEM_PROMPT_HTML_AESTHETIC, GAME_EXTRA_PROMPT
from util.utils_webagent import (
    get_web_element_rect,
    encode_image,
    extract_information,
    clip_message_and_obs,
    summarize_results,
    print_message,
)


# ------------------------
# Utility helpers
# ------------------------

def setup_logger(folder_path: str, base_dir: str) -> str:
    """Configure logger for both temp folder and independent log file."""
    os.makedirs(folder_path, exist_ok=True)
    log_file_path = os.path.join(folder_path, "agent.log")
    
    # Create independent log file outside temp folder (fixed filename)
    independent_log_dir = os.path.join(os.getcwd(), "webagent_logs")
    os.makedirs(independent_log_dir, exist_ok=True)
    independent_log_path = os.path.join(independent_log_dir, "webagent_detailed.log")
    
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass
    
    # Formatter with timestamp for better debugging
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    
    # Handler for temp folder log
    temp_handler = logging.FileHandler(log_file_path)
    temp_handler.setFormatter(formatter)
    logger.addHandler(temp_handler)
    
    # Handler for independent log file (append mode)
    independent_handler = logging.FileHandler(independent_log_path, mode='a')
    independent_handler.setFormatter(formatter)
    logger.addHandler(independent_handler)
    
    logger.setLevel(logging.INFO)
    
    # Log session separator and initial information
    logger.info(f"\n{'='*80}")
    logger.info(f"=== NEW WEBAGENT SESSION STARTED ===")
    logger.info(f"Session timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Temporary files directory: {base_dir}")
    logger.info(f"Independent log file: {independent_log_path}")
    logger.info(f"{'='*80}")
    
    return independent_log_path


def handle_alert_safe(driver: webdriver.Chrome, timeout: int = 3):
    """Safely handle any alert that might be present."""
    try:
        WebDriverWait(driver, timeout).until(EC.alert_is_present())
        alert = driver.switch_to.alert
        alert_text = alert.text
        logging.info(f"Alert detected with text: {alert_text}")
        alert.accept()
        return True, alert_text
    except (TimeoutException, NoAlertPresentException):
        return False, None
    except Exception as e:
        logging.warning(f"Error handling alert: {e}")
        return False, None


def safe_element_access(web_eles, index, action_type: str = "access"):
    """Safely access element from list with bounds checking."""
    try:
        index = int(index)
        if index < 0 or index >= len(web_eles):
            logging.error(
                f"Index {index} out of range for web_eles (length: {len(web_eles)})"
            )
            return None
        return web_eles[index]
    except (ValueError, TypeError) as e:
        logging.error(f"Invalid index for {action_type}: {index}, error: {e}")
        return None


@contextmanager
def create_driver_with_retry(options: webdriver.ChromeOptions, max_retries: int = 10, retry_delay: int = 2):
    """Create WebDriver with retry and always quit in finally."""
    driver = None
    last_exception = None
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                time.sleep(retry_delay + random.uniform(0, 1))
            driver = webdriver.Chrome(options=options)
            break
        except Exception as e:
            last_exception = e
            logging.warning(f"Attempt {attempt + 1} failed to create driver: {e}")
    if driver is None:
        raise last_exception if last_exception else RuntimeError("Failed to create WebDriver after retries")
    try:
        yield driver
    finally:
        try:
            driver.quit()
        except Exception as e:
            logging.debug(f"Error when quitting driver: {e}")


def cleanup_temp_resources(base_dir: str) -> None:
    try:
        if os.path.exists(base_dir):
            for attempt in range(3):
                try:
                    shutil.rmtree(base_dir, ignore_errors=True)
                    break
                except Exception as e:
                    if attempt < 2:
                        time.sleep(0.5)
                        continue
                    logging.warning(f"Failed to clean up {base_dir}: {e}")
    except Exception as e:
        logging.warning(f"Error during cleanup: {e}")


def _suppress_dialogs(driver: webdriver.Chrome) -> None:
    """Override alert/confirm/prompt to no-op on the current page context."""
    try:
        driver.execute_script(
            """
            try {
              window.alert = function(){};
              window.confirm = function(){ return true; };
              window.prompt = function(){ return ''; };
            } catch (e) {}
            """
        )
    except Exception:
        pass


def format_msg(it: int, init_msg: str, warn_obs: str, web_img_b64: str, web_text: str):
    """Compose the user message content for GPT with an image and optional text info."""
    if it == 2:
        init_msg = f"""
        In this iteration, do not eager to interact with the webpage. Think thoroughly about **ALL the ways of interactions** with the webpage based on the webpage screenshot. You need to analyze all the possibly interactable elements of the webpage, and plan to interact with them all in a proper order. For example, if there are some buttons, you should add \"click them all\" to your plan list. If there is a textbox, you should add \"type in the textbox\" to your plan list, etc. 

        There are some rules for creating the planned list:
        1. Analyze the webpage carefully and thoroughly to find **ALL the interactable elements** on the webpage. You may analyze the webpage from the top to the bottom, and from the left to the right, to find if there are any interactable elements to add to the planned list. Also, you can analyze the elements according to the given accessibility tree and the tags number of each element in the attached screenshot. DO NOT MISS ANY INTERACTABLE ELEMENTS! 
        2. For EVERY interactable element, think about one proper way to interact, and add it to the planned list.
        3. DO NOT plan homogenous interactions of same type of elements. For example, there are multiple buttons which belongs to the same choice, pick one of them. 
        4. DOUBLE CHECK whether you have missed any interactable elements. You should add any interactable elements in the webpage to the planned list.

        The item of the planned list should be like this: [INTERACTION_ID] [Numerical_Label] [Interaction_Type] [Interaction_ELEMENT] [BRIEF_DESCRIPTION_OF_THE_INTERACTION]. Do not describle interactions of a same element in different items of the list.**Output your planned interations in a json format in your thought.**

        I've provided the tag name of each element and the text it contains (if text exists). Note that <textarea> or <input> may be textbox, but not exactly. Please focus more on the screenshot and then refer to the textual information. \n{web_text}
        """
        init_msg_format = {
            "role": "user",
            "content": [
                {"type": "text", "text": init_msg},
            ],
        }
        init_msg_format["content"].append(
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{web_img_b64}"}}
        )
        return init_msg_format
    elif it == 3:
        init_msg = f"""
        Please double check your plan list, and find out whether you have missed ANY interactable elements. You should add any interactable elements in the webpage to the planned list. Because the total times of iterations you have is **limited**, **rank your plan list in an importance order from high to low**. (The elements you think are more important should interact earlier than those less important.) Then, if it is more than 5 interactions, **reduce your plan list to the top 5 most important interactions**. Output your plan list again in a json format in your thought. \n{web_text}
        """
        init_msg_format = {
            "role": "user",
            "content": [
                {"type": "text", "text": init_msg},
            ],
        }
        init_msg_format["content"].append(
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{web_img_b64}"}}
        )
        return init_msg_format
    elif it == 4:
        init_msg = (
            f"Now you have already planned all your interations, do them in order. I've provided the tag name of each element and the text it contains (if text exists). Note that <textarea> or <input> may be textbox, but not exactly. Please focus more on the screenshot and then refer to the textual information. \n{web_text}"
        )
        init_msg_format = {
            "role": "user",
            "content": [
                {"type": "text", "text": init_msg},
            ],
        }
        init_msg_format["content"].append(
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{web_img_b64}"}}
        )
        return init_msg_format
    else:
        curr_msg = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Observation:{warn_obs} please analyze the attached screenshot and give the Thought and Action. I've provided the tag name of each element and the text it contains (if text exists). Note that <textarea> or <input> may be textbox, but not exactly. Please focus more on the screenshot and then refer to the textual information.\n{web_text}",
                },
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{web_img_b64}"}},
            ],
        }
        return curr_msg


def call_gpt4v_api(temperature: float, openai: AzureOpenai, messages, tab_index=None):
    retry_times = 0
    openai_client = openai.client
    model = openai.model
    tab_info = f" (Tab {tab_index})" if tab_index is not None else ""
    
    while True:
        try:
            openai_response = openai_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=10240,
                temperature=temperature,
            )
            prompt_tokens = openai_response.usage.prompt_tokens
            completion_tokens = openai_response.usage.completion_tokens
            gpt_call_error = False
            # logging.info(f"GPT API call successful{tab_info}. Tokens: {prompt_tokens} prompt, {completion_tokens} completion")
            return prompt_tokens, completion_tokens, gpt_call_error, openai_response
        except Exception as e:
            error_details = {
                'error_type': type(e).__name__,
                'error_message': str(e),
                'retry_attempt': retry_times + 1,
                'tab': tab_index
            }
            logging.error(f"GPT API call failed{tab_info}: {error_details}")
            
            if type(e).__name__ in {"RateLimitError", "APIError"}:
                logging.warning(f"Rate limit or API error{tab_info}, waiting 2 seconds before retry")
                time.sleep(5)
            elif type(e).__name__ == "InvalidRequestError":
                logging.error(f"Invalid request error{tab_info}, cannot retry: {str(e)}")
                gpt_call_error = True
                return None, None, gpt_call_error, error_details
            else:
                logging.error(f"Unexpected API error{tab_info}, cannot retry: {str(e)}")
                gpt_call_error = True
                return None, None, gpt_call_error, error_details
        
        retry_times += 1
        if retry_times == 10:
            error_details = {
                'error_type': 'RetryLimitExceeded',
                'error_message': f'Failed after {retry_times} attempts',
                'retry_attempt': retry_times,
                'tab': tab_index
            }
            logging.error(f"GPT API call failed{tab_info} after {retry_times} retries")
            return None, None, True, error_details


def exec_action_click(info, web_ele, driver_task: webdriver.Chrome):
    # Handle alerts first
    handle_alert_safe(driver_task, timeout=1)
    try:
        driver_task.execute_script("arguments[0].setAttribute('target', '_self')", web_ele)
        web_ele.click()
        time.sleep(0.5)
        handle_alert_safe(driver_task, timeout=1)
    except UnexpectedAlertPresentException:
        logging.info("Alert present during click, handling it")
        handle_alert_safe(driver_task)
        try:
            web_ele.click()
        except Exception:
            logging.warning("Failed to retry click after handling alert")


def exec_action_type(info, web_ele, driver_task: webdriver.Chrome):
    # Handle alerts first
    handle_alert_safe(driver_task, timeout=1)
    warn_obs = ""
    type_content = info["content"]
    ele_tag_name = web_ele.tag_name.lower()
    ele_type = web_ele.get_attribute("type")
    if (ele_tag_name != "input" and ele_tag_name != "textarea") or (
        ele_tag_name == "input" and ele_type not in ["text", "search", "password", "email", "tel"]
    ):
        warn_obs = (
            f"note: The web element you're trying to type may not be a textbox, and its tag name is <{web_ele.tag_name}>, type is {ele_type}."
        )
    try:
        # Try to clear first
        web_ele.clear()
        # Fallback delete
        try:
            web_ele.send_keys(Keys.CONTROL + "a")
        except Exception:
            pass
        web_ele.send_keys(" ")
        web_ele.send_keys(Keys.BACKSPACE)
    except UnexpectedAlertPresentException:
        logging.info("Alert present during text clearing, handling it")
        handle_alert_safe(driver_task)
    except Exception:
        pass

    try:
        actions = ActionChains(driver_task)
        actions.click(web_ele).perform()
        actions.pause(0.5)
        try:
            driver_task.execute_script(
                """
                window.onkeydown = function(e) {
                  if (e.keyCode == 32 && e.target.type != 'text' && e.target.type != 'textarea' && e.target.type != 'search') {
                    e.preventDefault();
                  }
                };
                """
            )
        except Exception:
            pass
        actions.send_keys(type_content)
        actions.pause(0.5)
        actions.send_keys(Keys.ENTER)
        actions.perform()
        time.sleep(0.5)
        handle_alert_safe(driver_task, timeout=1)
    except UnexpectedAlertPresentException:
        logging.info("Alert present during typing, handling it")
        handle_alert_safe(driver_task)
    return warn_obs


def exec_action_scroll(info, web_eles, driver_task: webdriver.Chrome, window_height: int):
    scroll_ele_number = info["number"]
    scroll_content = info["content"]
    if scroll_ele_number == "WINDOW":
        if scroll_content == "down":
            driver_task.execute_script(f"window.scrollBy(0, {window_height * 2 // 3});")
        else:
            driver_task.execute_script(f"window.scrollBy(0, {-window_height * 2 // 3});")
    else:
        scroll_ele_number = int(scroll_ele_number)
        web_ele = web_eles[scroll_ele_number]
        actions = ActionChains(driver_task)
        driver_task.execute_script("arguments[0].focus();", web_ele)
        if scroll_content == "down":
            actions.key_down(Keys.ALT).send_keys(Keys.ARROW_DOWN).key_up(Keys.ALT).perform()
        else:
            actions.key_down(Keys.ALT).send_keys(Keys.ARROW_UP).key_up(Keys.ALT).perform()
    time.sleep(0.5)


class _TabContext:
    """Per-tab state container."""

    def __init__(self, index: int, handle: str, tab_dir: str, html_path: str, instruction: str):
        self.index = index
        self.handle = handle
        self.tab_dir = tab_dir
        self.html_path = html_path
        self.instruction = instruction
        self.messages = []
        self.iteration = 0
        self.obs_prompt = ""
        self.init_msg_text = ""
        self.system_message = None


class BrowserTabsBatchRunner:
    """Run multiple HTML evaluations in one Chrome instance with multiple tabs.

    Browser operations are serialized by a lock; GPT calls run in parallel.
    """

    def __init__(
        self,
        html_list,
        instructions,
        temperature: float = 0.0,
        max_iter: int = 7,
        api_key: str = "key",
        api_model: str = "gpt-4-vision-preview",
        max_attached_imgs: int = 30,
        window_width: int = 1920,
        window_height: int = 1200,
        fix_box_color: bool = False,
        API_INFO=API_INFOS,
        timeout_seconds: int = 600,
        model_name: str = "gpt-4-vision-preview",
    ):
        self.html_list = list(html_list)
        # Normalize instructions to a list matching html_list length
        if isinstance(instructions, (list, tuple)):
            if len(instructions) != len(self.html_list):
                raise ValueError("Length of instructions must match length of html_list")
            self.instructions = list(instructions)
        else:
            # Broadcast single instruction to all tabs
            self.instructions = [instructions] * len(self.html_list)
        self.temperature = temperature
        self.max_iter = max_iter
        self.api_key = api_key
        self.api_model = api_model
        self.max_attached_imgs = max_attached_imgs
        self.window_width = window_width
        self.window_height = window_height
        self.fix_box_color = fix_box_color
        self.API_INFO = API_INFO
        self.timeout_seconds = timeout_seconds
        self.model_name = model_name
        # Run-scoped resources
        self.thread_id = threading.current_thread().ident
        self.unique_id = str(uuid.uuid4())[:8]
        self.timestamp_ns = int(time.time_ns())
        self.base_dir = os.path.abspath(
            f"./webagent_tmp_batch/{self.model_name}/{self.timestamp_ns}_{self.thread_id}_{self.unique_id}"
        )
        os.makedirs(self.base_dir, exist_ok=True)

        self.independent_log_path = setup_logger(self.base_dir, self.base_dir)

        # Shared driver and lock
        self.driver = None
        self.driver_lock = threading.Lock()
        self.print_lock = threading.Lock()

        # Per tab state
        self.tabs = []  # list[_TabContext]

        # System prompt is decided per tab (per-instruction)
        self.system_message = None  # not used globally anymore

    def _build_options(self) -> webdriver.ChromeOptions:
        options = webdriver.ChromeOptions()
        # Use new headless for better WebGL support on recent Chrome
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-plugins")
        options.add_argument("--disable-javascript-harmony-shipping")
        options.add_argument("--disable-background-timer-throttling")
        options.add_argument("--disable-backgrounding-occluded-windows")
        options.add_argument("--disable-renderer-backgrounding")
        options.add_argument("--disable-features=TranslateUI")
        options.add_argument("--disable-ipc-flooding-protection")
        # Enable WebGL via software renderer (SwiftShader) in headless environments
        options.add_argument("--ignore-gpu-blocklist")
        options.add_argument("--enable-webgl")
        options.add_argument("--use-gl=swiftshader")
        # Auto-accept unexpected alerts to avoid command failures
        try:
            # Use widely supported value
            options.set_capability("unhandledPromptBehavior", "accept")
        except Exception:
            # Safe fallback; some drivers may not accept the capability call
            pass
        options.add_argument(
            "--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        )
        # Shared profile for the batch run
        user_data_dir = os.path.join(self.base_dir, "chrome_user_data")
        options.add_argument(f"--user-data-dir={user_data_dir}")
        # Random remote debugging port to avoid conflicts
        debug_port = random.randint(9222, 29999)
        options.add_argument(f"--remote-debugging-port={debug_port}")
        return options

    def _prepare_tabs(self, driver: webdriver.Chrome) -> None:
        # Write html files to per-tab directories and open in tabs
        for idx, html in enumerate(self.html_list):
            tab_dir = os.path.join(self.base_dir, f"tab_{idx:02d}")
            os.makedirs(tab_dir, exist_ok=True)
            html_tmp_file = os.path.abspath(os.path.join(tab_dir, "tmp.html"))
            with open(html_tmp_file, "w", encoding="utf-8") as f:
                f.write(html)
            url = f"file://{html_tmp_file}"
            if idx == 0:
                # Load first page in the initial tab with alert handling
                for attempt in range(3):
                    try:
                        driver.get(url)
                        # Inject dialog suppression as soon as possible after successful navigation
                        _suppress_dialogs(driver)
                        break
                    except UnexpectedAlertPresentException:
                        handle_alert_safe(driver, timeout=1)
                        time.sleep(0.2)
                handle = driver.current_window_handle
            else:
                # Create new tab with retries and alert handling
                new_handle = None
                for attempt in range(5):
                    try:
                        driver.switch_to.new_window("tab")
                        new_handle = driver.current_window_handle
                        break
                    except UnexpectedAlertPresentException:
                        handle_alert_safe(driver, timeout=1)
                        time.sleep(0.2)
                if new_handle is None:
                    # Fallback: open via window.open
                    try:
                        driver.execute_script("window.open('about:blank','_blank');")
                        time.sleep(0.2)
                        new_handle = [h for h in driver.window_handles if h != driver.current_window_handle][-1]
                    except Exception:
                        pass
                if new_handle is None:
                    raise RuntimeError("Failed to create a new tab after retries")
                driver.switch_to.window(new_handle)
                # Navigate to the URL with retries
                for attempt in range(3):
                    try:
                        driver.get(url)
                        # Inject dialog suppression after successful navigation
                        _suppress_dialogs(driver)
                        break
                    except UnexpectedAlertPresentException:
                        handle_alert_safe(driver, timeout=1)
                        time.sleep(0.2)
                handle = driver.current_window_handle
            self.tabs.append(
                _TabContext(
                    index=idx,
                    handle=handle,
                    tab_dir=tab_dir,
                    html_path=html_tmp_file,
                    instruction=self.instructions[idx],
                )
            )

        # Initialize each tab minimally (focus body, set key handlers)
        for tab in self.tabs:
            with self.driver_lock:
                driver.switch_to.window(tab.handle)
                # Handle any initial alert immediately after load (e.g., WebGL unsupported)
                try:
                    handled, _ = handle_alert_safe(driver, timeout=2)
                    if handled:
                        time.sleep(0.1)
                except Exception:
                    pass
                # Re-apply dialog suppression at init
                _suppress_dialogs(driver)
                try:
                    driver.set_window_size(self.window_width, self.window_height)
                except WebDriverException:
                    pass
                try:
                    driver.find_element(By.TAG_NAME, "body").click()
                except Exception:
                    pass
                try:
                    driver.execute_script(
                        """
                        window.onkeydown = function(e) {
                          if (e.keyCode == 32 && e.target.type != 'text' && e.target.type != 'textarea') {
                            e.preventDefault();
                          }
                        };
                        """
                    )
                except Exception:
                    pass

            # Initialize messages per tab (system prompt depends on instruction)
            system_prompt = SYSTEM_PROMPT_HTML_AESTHETIC
            if "Game dev" in tab.instruction:
                system_prompt = system_prompt.replace("<GAME_EXTRA_PROMPT>", GAME_EXTRA_PROMPT)
            else:
                system_prompt = system_prompt.replace("<GAME_EXTRA_PROMPT>", "")
            tab.system_message = {"role": "system", "content": system_prompt}

            init_msg = (
                f"Now you have a webpage evaluation task: {tab.instruction}  Please interact with https://www.example.com and evaluate the webpage. \n"
            )
            init_msg = init_msg.replace("https://www.example.com", f"file://{tab.html_path}")
            tab.messages = [tab.system_message]
            tab.obs_prompt = "Observation: please analyze the attached screenshot and give the Thought and Action. "
            tab.init_msg_text = init_msg

    def _run_single_tab(self, tab: _TabContext):
        start_info = {
            'tab_index': tab.index,
            'tab_dir': tab.tab_dir,
            'html_path': tab.html_path,
            'instruction': tab.instruction[:200],
            'base_dir': self.base_dir,
            'timestamp': time.time()
        }
        logging.info(f"Starting Tab {tab.index} | Details: {start_info}")
        
        client = AzureOpenai(apis=self.API_INFO)
        it = 0
        accumulate_prompt_token = 0
        accumulate_completion_token = 0
        fail_obs = ""
        warn_obs = ""
        pattern = r"Thought:|Action:|Observation:"

        try:
            while it < self.max_iter:
                if not it == 0 and not it == 1:
                    tab.init_msg_text = tab.init_msg_text + tab.obs_prompt
                it += 1
                if it == 1:
                    it = 2

                # Browser phase: annotate elements and take screenshot
                if not fail_obs:
                    # Acquire lock only for browser operations
                    for _ in range(10):
                        try:
                            with self.driver_lock:
                                self.driver.switch_to.window(tab.handle)
                                handle_alert_safe(self.driver, timeout=1)
                                # Re-apply dialog suppression each iteration
                                _suppress_dialogs(self.driver)
                                rects, web_eles, web_eles_text = get_web_element_rect(
                                    self.driver, fix_color=self.fix_box_color
                                )
                                img_path = os.path.join(tab.tab_dir, f"screenshot{it}.png")
                                self.driver.save_screenshot(img_path)
                            break
                        except Exception as e:
                            logging.error("Driver error when adding set-of-mark.")
                            logging.error(e)
                            time.sleep(0.3)
                    b64_img = encode_image(img_path)
                    curr_msg = format_msg(it, tab.init_msg_text, warn_obs, b64_img, web_eles_text)
                    tab.messages.append(curr_msg)
                else:
                    curr_msg = {"role": "user", "content": fail_obs}
                    tab.messages.append(curr_msg)

                # Clip to avoid too many images
                tab.messages = clip_message_and_obs(tab.messages, self.max_attached_imgs)

                # Model phase: call GPT without holding lock
                prompt_tokens, completion_tokens, gpt_call_error, api_result = call_gpt4v_api(
                    self.temperature, client, tab.messages, tab.index
                )
                if gpt_call_error:
                    error_msg = f"❌ GPT API call error - Tab {tab.index}: Failed due to API error\n   📁 Temp folder: {tab.tab_dir}"
                    print(error_msg)  # Command line output
                    logging.error(error_msg)
                    error_info = {
                        'tab_index': tab.index,
                        'tab_dir': tab.tab_dir,
                        'html_path': tab.html_path,
                        'instruction': tab.instruction[:200],
                        'iteration': it,
                        'base_dir': self.base_dir
                    }
                    if isinstance(api_result, dict):
                        logging.error(f"GPT API error details - Tab {tab.index}: {api_result} | Context: {error_info}")
                    else:
                        logging.error(f"GPT API call failed - Tab {tab.index}: Unknown error | Context: {error_info}")
                    return None  # GPT API failure returns None directly, no further processing
                else:
                    openai_response = api_result
                    accumulate_prompt_token += prompt_tokens
                    accumulate_completion_token += completion_tokens

                gpt_4v_res = openai_response.choices[0].message.content
                tab.messages.append({"role": "assistant", "content": gpt_4v_res})

                # Remove the rectangles before potential actions
                try:
                    with self.driver_lock:
                        self.driver.switch_to.window(tab.handle)
                        if 'rects' in locals() and rects:
                            for rect_ele in rects:
                                try:
                                    self.driver.execute_script("arguments[0].remove()", rect_ele)
                                except Exception:
                                    pass
                            rects = []
                except Exception:
                    pass

                if it in (1, 2, 3):
                    # Planning-only iterations
                    continue
                elif "FINISH" in gpt_4v_res:
                    break
                else:
                    try:
                        assert "Thought:" in gpt_4v_res and "Action:" in gpt_4v_res
                    except AssertionError as e:
                        logging.error(e)
                        fail_obs = "Format ERROR: Both 'Thought' and 'Action' should be included in your reply."
                        continue

                chosen_action = re.split(pattern, gpt_4v_res)[2].strip()
                action_key, info = extract_information(chosen_action)
                fail_obs = ""
                warn_obs = ""

                # Execute action under lock on the correct tab
                try:
                    with self.driver_lock:
                        handle_alert_safe(self.driver, timeout=1)
                        self.driver.switch_to.window(tab.handle)

                        if action_key == "click":
                            click_ele_number = int(info[0])
                            web_ele = safe_element_access(web_eles, click_ele_number, "click")
                            if web_ele is None:
                                raise IndexError(
                                    f"Click element index {click_ele_number} is out of range"
                                )
                            ele_tag_name = web_ele.tag_name.lower()
                            ele_type = web_ele.get_attribute("type")
                            exec_action_click(info, web_ele, self.driver)
                            if ele_tag_name == "button" and ele_type == "submit":
                                time.sleep(0.5)

                        elif action_key == "wait":
                            time.sleep(0.5)

                        elif action_key == "type":
                            type_ele_number = int(info["number"]) if isinstance(info, dict) else int(info[0])
                            web_ele = safe_element_access(web_eles, type_ele_number, "type")
                            if web_ele is None:
                                raise IndexError(
                                    f"Type element index {type_ele_number} is out of range"
                                )
                            warn_obs = exec_action_type(info, web_ele, self.driver)

                        elif action_key == "scroll":
                            # Scroll window or specific element
                            exec_action_scroll(info, web_eles, self.driver, self.window_height)

                        elif action_key == "up":
                            ActionChains(self.driver).send_keys(Keys.ARROW_UP).perform()
                            time.sleep(0.5)

                        elif action_key == "down":
                            ActionChains(self.driver).send_keys(Keys.ARROW_DOWN).perform()
                            time.sleep(0.5)

                        elif action_key == "left":
                            ActionChains(self.driver).send_keys(Keys.ARROW_LEFT).perform()
                            time.sleep(0.5)

                        elif action_key == "right":
                            ActionChains(self.driver).send_keys(Keys.ARROW_RIGHT).perform()
                            time.sleep(0.5)

                        elif action_key == "FINISH":
                            break
                        else:
                            # Not implemented action kind
                            fail_obs = (
                                "The action you have chosen is not supported. Please adjust and try again."
                            )
                            time.sleep(0.5)
                            continue

                        fail_obs = ""
                except UnexpectedAlertPresentException:
                    logging.info("Alert present during action execution, handling it")
                    with self.driver_lock:
                        alert_handled, alert_text = handle_alert_safe(self.driver)
                    if alert_handled:
                        logging.info(f"Alert handled: {alert_text}")
                        fail_obs = ""
                    else:
                        fail_obs = "Alert handling failed. Please try a different action."
                except Exception as e:
                    logging.error("driver error info:")
                    logging.error(e)
                    if "element click intercepted" in str(e).lower():
                        fail_obs = ""
                    elif "out of range" in str(e).lower() or "index" in str(e).lower():
                        fail_obs = (
                            f"Element index error: {e}. Please check the available elements and try again."
                        )
                    else:
                        fail_obs = (
                            "The action you have chosen cannot be executed. Please double-check if you have selected the wrong Numerical Label or Action or Action format. Then provide the revised Thought and Action."
                        )
                    time.sleep(1.5)

            # Summarize
            agent_avg_score, score_list = summarize_results(tab.messages)
            print_message(tab.messages, tab.tab_dir)
            
     

            assistant_texts = []
            for m in tab.messages:
                if m.get("role") == "assistant":
                    content = m.get("content")
                    if isinstance(content, str):
                        assistant_texts.append(content)
            output = [
                f"=== Assistant messages for tab {tab.index} ===",
            ] + assistant_texts + ["=== End of assistant messages ==="]

            #####################################
            # Need to log the output for debug! #
            #####################################
            
            # Log the output information for debugging
            logging.info("Output information from line 760:")
            for line in output:
                logging.info(line)

            # with self.print_lock:
                # print("\n".join(output))

            return score_list if score_list else [0]



            # for debug, use sum of score
            # return sum(score_list) if score_list else 0
        

        except Exception as e:
            error_msg = f"❌ Tab {tab.index} critical error: {str(e)} - Returning failure status\n   📁 Temp folder: {tab.tab_dir}"
            print(error_msg)
            
            logging.error(error_msg)
            
            # Detailed error information logging
            error_context = {
                'tab_index': tab.index,
                'tab_dir': tab.tab_dir,
                'html_path': tab.html_path,
                'instruction': tab.instruction[:200],
                'base_dir': self.base_dir,
                'exception_type': type(e).__name__,
                'exception_message': str(e)
            }
            
            logging.error(f"Critical error in tab run: {e} | Context: {error_context}")
            return None  # Return None indicates run failure, distinct from normal score of 0

    def run(self):
        batch_info = {
            'batch_id': f"{self.timestamp_ns}_{self.thread_id}_{self.unique_id}",
            'html_count': len(self.html_list),
            'base_dir': self.base_dir,
            'independent_log_path': self.independent_log_path,
            'thread_id': self.thread_id,
            'timestamp': self.timestamp_ns,
            'max_iter': self.max_iter,
            'timeout_seconds': self.timeout_seconds,
            'window_size': f"{self.window_width}x{self.window_height}"
        }

        
        options = self._build_options()
        try:
            with create_driver_with_retry(options, max_retries=10, retry_delay=2) as driver:
                self.driver = driver
                # Prepare tabs
                self._prepare_tabs(driver)
                logging.info(f"Successfully prepared {len(self.tabs)} tabs")

                # Execute per tab in parallel (model calls dominate time; browser ops are short and locked)
                results = [None] * len(self.tabs)
                with ThreadPoolExecutor(max_workers=len(self.tabs)) as executor:
                    futures = [
                        executor.submit(self._run_single_tab, tab) for tab in self.tabs
                    ]
                    for i, fut in enumerate(futures):
                        try:
                            results[i] = fut.result(timeout=self.timeout_seconds)
                        except FuturesTimeoutError:
                            tab_dir = self.tabs[i].tab_dir if i < len(self.tabs) else "unknown"
                            error_msg = f"❌ Tab {i} timeout ({self.timeout_seconds}s) - Returning failure status\n   📁 Temp folder: {tab_dir}"
                            print(error_msg)
                            
                            # Also log user-friendly error information
                            logging.warning(error_msg)
                            
                            # Detailed timeout information logging
                            timeout_context = {
                                'tab_index': i,
                                'tab_dir': tab_dir,
                                'timeout_seconds': self.timeout_seconds,
                                'base_dir': self.base_dir,
                                'instruction': self.tabs[i].instruction[:200] if i < len(self.tabs) else "unknown"
                            }
                            logging.warning(f"Tab {i} timed out after {self.timeout_seconds} seconds | Context: {timeout_context}")
                            # Try to cancel future and close browser to terminate thread
                            try:
                                fut.cancel()
                            except Exception:
                                pass

                            # Close browser driver to forcibly interrupt running tab threads
                            try:
                                with self.driver_lock:
                                    if self.driver:
                                        self.driver.quit()
                            except Exception:
                                pass

                            # Set driver reference to None to prevent subsequent reuse
                            self.driver = None

                            # Mark result as None
                            results[i] = None  # Timeout returns None to indicate failure
                        except Exception as e:
                            tab_dir = self.tabs[i].tab_dir if i < len(self.tabs) else "unknown"
                            error_msg = f"❌ Tab {i} runtime error: {str(e)} - Returning failure status\n   📁 Temp folder: {tab_dir}"
                            print(error_msg)
                            
                            # Also log user-friendly error information
                            logging.error(error_msg)
                            
                            # Detailed exception information logging
                            exception_context = {
                                'tab_index': i,
                                'tab_dir': tab_dir,
                                'base_dir': self.base_dir,
                                'exception_type': type(e).__name__,
                                'exception_message': str(e),
                                'instruction': self.tabs[i].instruction[:200] if i < len(self.tabs) else "unknown"
                            }
                            logging.error(f"Error in tab {i}: {e} | Context: {exception_context}")
                            results[i] = None  # Exception returns None to indicate failure
                
                # Count results and log (distinguish between normal completion and run failure)
                success_count = sum(1 for r in results if r is not None)  # Normal completion (including 0 score)
                failed_count = sum(1 for r in results if r is None)  # Run failure
                valid_scores = [r for r in results if r is not None]
                
                # Detailed completion statistics
                completion_info = {
                    'batch_id': f"{self.timestamp_ns}_{self.thread_id}_{self.unique_id}",
                    'total_tabs': len(results),
                    'success_count': success_count,
                    'failed_count': failed_count,
                    # 'avg_score': round(avg_score, 2),
                    'base_dir': self.base_dir,
                    'all_scores': results,
                    'completion_time': time.time()
                }
                
                logging.info(f"=== Batch evaluation completed === | Statistics: {completion_info}")
                print(f"✅ Evaluation completed (Batch ID: {completion_info['batch_id'][:12]}...) - Success: {success_count}/{len(results)}, Failed: {failed_count}")
                
                results = [x if x is not None else [0.0] for x in results]
                return results
        finally:
            
            logging.info(f"{'='*80}\n")


def judge_website_batch(
    html_list,
    instructions,
    temperature: float = 0.0,
    max_iter: int = 7,
    api_key: str = "key",
    api_model: str = "gpt-4-vision-preview",
    max_attached_imgs: int = 30,
    window_width: int = 1920,
    window_height: int = 1200,
    fix_box_color: bool = False,
    API_INFO=API_INFOS,
    timeout_seconds: int = 600,
    model_name: str = "gpt-4-vision-preview",
):
    """Batch judge multiple HTML strings using one Chrome instance with multiple tabs.

    Returns a list of scores: [score1, score2, ...] where:
    - score is a float (including 0.0 for normal zero scores)
    - None indicates API errors, timeouts, or other failures
    """
    runner = BrowserTabsBatchRunner(
        html_list=html_list,
        instructions=instructions,
        temperature=temperature,
        max_iter=max_iter,
        api_key=api_key,
        api_model=api_model,
        max_attached_imgs=max_attached_imgs,
        window_width=window_width,
        window_height=window_height,
        fix_box_color=fix_box_color,
        API_INFO=API_INFO,
        timeout_seconds=timeout_seconds,
        model_name=model_name,
    )
    return runner.run()


def judge_website(
    html: str,
    instruction: str,
    temperature: float = 0.0,
    max_iter: int = 7,
    api_key: str = "key",
    api_model: str = "gpt-4-vision-preview",
    max_attached_imgs: int = 30,
    window_width: int = 1920,
    window_height: int = 1200,
    fix_box_color: bool = False,
    API_INFO=API_INFOS,
    timeout_seconds: int = 600,
):
    """Compatibility wrapper to judge a single HTML using the batch pipeline."""
    results = judge_website_batch(
        [html],
        [instruction],
        temperature,
        max_iter,
        api_key,
        api_model,
        max_attached_imgs,
        window_width,
        window_height,
        fix_box_color,
        API_INFO,
        timeout_seconds,
    )
    return results[0]

