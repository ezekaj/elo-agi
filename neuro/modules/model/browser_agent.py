"""
Browser Agent - AI that can browse and interact with websites

Uses Playwright/Selenium to:
1. Open websites
2. Read content
3. Click buttons
4. Fill forms
5. Take screenshots
6. Learn from web pages

Ported from AGIELO for NEURO AGI v0.9
"""

import time
import re
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

# Try different browser automation libraries
BROWSER_AVAILABLE = False
BROWSER_TYPE = None

try:
    from playwright.sync_api import sync_playwright

    BROWSER_AVAILABLE = True
    BROWSER_TYPE = "playwright"
except ImportError:
    pass

if not BROWSER_AVAILABLE:
    try:
        from selenium import webdriver
        from selenium.webdriver.common.by import By  # noqa: F401
        from selenium.webdriver.chrome.options import Options

        BROWSER_AVAILABLE = True
        BROWSER_TYPE = "selenium"
    except ImportError:
        pass


class BrowserAgent:
    """
    AI agent that can browse and interact with websites.

    Capabilities:
    - Navigate to URLs
    - Read page content
    - Click elements
    - Fill forms
    - Scroll pages
    - Take screenshots
    - Extract information
    """

    def __init__(self, headless: bool = True):
        """
        Initialize browser agent.

        Args:
            headless: Run browser without GUI (faster)
        """
        self.headless = headless
        self.browser = None
        self.page = None
        self.playwright = None
        self.context = None
        self.history: List[Dict] = []
        self.screenshots_dir = Path.home() / ".neuro" / "screenshots"
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)

    @property
    def is_available(self) -> bool:
        """Check if browser automation is available."""
        return BROWSER_AVAILABLE

    @property
    def backend(self) -> Optional[str]:
        """Get the browser backend type."""
        return BROWSER_TYPE

    def start(self) -> bool:
        """Start the browser."""
        if not BROWSER_AVAILABLE:
            return False

        if BROWSER_TYPE == "playwright":
            return self._start_playwright()
        elif BROWSER_TYPE == "selenium":
            return self._start_selenium()

        return False

    def _start_playwright(self) -> bool:
        """Start Playwright browser."""
        try:
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.chromium.launch(headless=self.headless)
            self.context = self.browser.new_context(
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            )
            self.page = self.context.new_page()
            return True
        except Exception as e:
            print(f"[Browser] Playwright error: {e}")
            return False

    def _start_selenium(self) -> bool:
        """Start Selenium browser."""
        try:
            options = Options()
            if self.headless:
                options.add_argument("--headless")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")

            from selenium.webdriver.chrome.service import Service
            from webdriver_manager.chrome import ChromeDriverManager

            service = Service(ChromeDriverManager().install())
            self.browser = webdriver.Chrome(service=service, options=options)
            self.page = self.browser
            return True
        except Exception as e:
            print(f"[Browser] Selenium error: {e}")
            return False

    def goto(self, url: str, timeout: int = 30000) -> Dict:
        """
        Navigate to a URL.

        Args:
            url: The URL to visit
            timeout: Timeout in milliseconds

        Returns:
            Dict with page info
        """
        if not self.page:
            if not self.start():
                return {"success": False, "error": "Browser not available"}

        # Ensure URL has protocol
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        try:
            if BROWSER_TYPE == "playwright":
                self.page.goto(url, timeout=timeout)
                title = self.page.title()
            else:
                self.browser.set_page_load_timeout(timeout // 1000)
                self.browser.get(url)
                title = self.browser.title

            self.history.append(
                {"action": "goto", "url": url, "title": title, "time": datetime.now().isoformat()}
            )

            return {"success": True, "url": url, "title": title}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_content(self, max_length: int = 5000) -> str:
        """
        Get the text content of the current page.

        Args:
            max_length: Maximum characters to return

        Returns:
            Text content of the page
        """
        if not self.page:
            return ""

        try:
            if BROWSER_TYPE == "playwright":
                content = self.page.content()
            else:
                content = self.browser.page_source

            # Extract text from HTML
            text = re.sub(r"<script[^>]*>.*?</script>", "", content, flags=re.DOTALL)
            text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()

            return text[:max_length]

        except Exception as e:
            return f"Error reading page: {e}"

    def get_title(self) -> str:
        """Get the current page title."""
        if not self.page:
            return ""

        try:
            if BROWSER_TYPE == "playwright":
                return self.page.title()
            else:
                return self.browser.title
        except Exception:
            return ""

    def get_url(self) -> str:
        """Get the current page URL."""
        if not self.page:
            return ""

        try:
            if BROWSER_TYPE == "playwright":
                return self.page.url
            else:
                return self.browser.current_url
        except Exception:
            return ""

    def click(self, selector: str, timeout: int = 5000) -> Dict:
        """
        Click an element on the page.

        Args:
            selector: CSS selector or text to click
            timeout: Timeout in milliseconds

        Returns:
            Dict with result
        """
        if not self.page:
            return {"success": False, "error": "No page loaded"}

        try:
            if BROWSER_TYPE == "playwright":
                # Try CSS selector first, then text
                try:
                    self.page.click(selector, timeout=timeout)
                except Exception:
                    self.page.click(f"text={selector}", timeout=timeout)
            else:
                from selenium.webdriver.common.by import By

                try:
                    element = self.browser.find_element(By.CSS_SELECTOR, selector)
                except Exception:
                    element = self.browser.find_element(By.LINK_TEXT, selector)
                element.click()

            self.history.append(
                {"action": "click", "selector": selector, "time": datetime.now().isoformat()}
            )

            return {"success": True, "clicked": selector}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def fill(self, selector: str, text: str) -> Dict:
        """
        Fill a form field.

        Args:
            selector: CSS selector for the input
            text: Text to enter

        Returns:
            Dict with result
        """
        if not self.page:
            return {"success": False, "error": "No page loaded"}

        try:
            if BROWSER_TYPE == "playwright":
                self.page.fill(selector, text)
            else:
                from selenium.webdriver.common.by import By

                element = self.browser.find_element(By.CSS_SELECTOR, selector)
                element.clear()
                element.send_keys(text)

            self.history.append(
                {
                    "action": "fill",
                    "selector": selector,
                    "text_length": len(text),
                    "time": datetime.now().isoformat(),
                }
            )

            return {"success": True, "filled": selector}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def type_text(self, selector: str, text: str, delay: int = 50) -> Dict:
        """
        Type text character by character (more human-like).

        Args:
            selector: CSS selector for the input
            text: Text to type
            delay: Delay between keystrokes in ms

        Returns:
            Dict with result
        """
        if not self.page:
            return {"success": False, "error": "No page loaded"}

        try:
            if BROWSER_TYPE == "playwright":
                self.page.type(selector, text, delay=delay)
            else:
                from selenium.webdriver.common.by import By

                element = self.browser.find_element(By.CSS_SELECTOR, selector)
                for char in text:
                    element.send_keys(char)
                    time.sleep(delay / 1000)

            return {"success": True, "typed": len(text)}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def scroll(self, direction: str = "down", amount: int = 500) -> Dict:
        """
        Scroll the page.

        Args:
            direction: "down" or "up"
            amount: Pixels to scroll

        Returns:
            Dict with result
        """
        if not self.page:
            return {"success": False, "error": "No page loaded"}

        try:
            scroll_amount = amount if direction == "down" else -amount

            if BROWSER_TYPE == "playwright":
                self.page.evaluate(f"window.scrollBy(0, {scroll_amount})")
            else:
                self.browser.execute_script(f"window.scrollBy(0, {scroll_amount})")

            return {"success": True, "scrolled": direction, "amount": amount}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def screenshot(self, name: str = None, full_page: bool = False) -> str:
        """
        Take a screenshot.

        Args:
            name: Screenshot filename (without extension)
            full_page: Capture full page (Playwright only)

        Returns:
            Path to the screenshot
        """
        if not self.page:
            return ""

        try:
            name = name or f"screenshot_{int(time.time())}"
            path = str(self.screenshots_dir / f"{name}.png")

            if BROWSER_TYPE == "playwright":
                self.page.screenshot(path=path, full_page=full_page)
            else:
                self.browser.save_screenshot(path)

            return path

        except Exception as e:
            return f"Error: {e}"

    def get_links(self, max_links: int = 20) -> List[Dict]:
        """
        Get all links on the page.

        Args:
            max_links: Maximum number of links to return

        Returns:
            List of link dicts with text and href
        """
        if not self.page:
            return []

        try:
            if BROWSER_TYPE == "playwright":
                links = self.page.eval_on_selector_all(
                    "a[href]",
                    "elements => elements.map(e => ({text: e.innerText.trim(), href: e.href}))",
                )
            else:
                from selenium.webdriver.common.by import By

                elements = self.browser.find_elements(By.TAG_NAME, "a")
                links = []
                for e in elements:
                    try:
                        links.append({"text": e.text.strip(), "href": e.get_attribute("href")})
                    except Exception:
                        pass

            # Filter valid links
            valid_links = [
                lnk for lnk in links if lnk.get("href") and lnk.get("text") and len(lnk["text"]) > 0
            ]

            return valid_links[:max_links]

        except Exception:
            return []

    def search_google(self, query: str, num_results: int = 10) -> List[Dict]:
        """
        Search Google and return results.

        Args:
            query: Search query
            num_results: Number of results to return

        Returns:
            List of search results
        """
        # URL encode the query
        from urllib.parse import quote_plus

        encoded_query = quote_plus(query)

        result = self.goto(f"https://www.google.com/search?q={encoded_query}")
        if not result.get("success"):
            return []

        time.sleep(2)  # Wait for results to load

        # Get links
        links = self.get_links(max_links=50)

        # Filter to actual search results (not Google internal links)
        results = []
        for link in links:
            href = link.get("href", "")
            if (
                href
                and "google.com" not in href
                and "/search?" not in href
                and href.startswith("http")
            ):
                results.append(link)

        return results[:num_results]

    def wait(self, milliseconds: int) -> None:
        """Wait for a specified time."""
        time.sleep(milliseconds / 1000)

    def wait_for_selector(self, selector: str, timeout: int = 10000) -> bool:
        """
        Wait for an element to appear.

        Args:
            selector: CSS selector
            timeout: Timeout in milliseconds

        Returns:
            True if element appeared, False otherwise
        """
        if not self.page:
            return False

        try:
            if BROWSER_TYPE == "playwright":
                self.page.wait_for_selector(selector, timeout=timeout)
            else:
                from selenium.webdriver.support.ui import WebDriverWait
                from selenium.webdriver.support import expected_conditions as EC
                from selenium.webdriver.common.by import By

                WebDriverWait(self.browser, timeout / 1000).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                )
            return True
        except Exception:
            return False

    def close(self) -> None:
        """Close the browser."""
        try:
            if BROWSER_TYPE == "playwright":
                if self.context:
                    self.context.close()
                if self.browser:
                    self.browser.close()
                if self.playwright:
                    self.playwright.stop()
            else:
                if self.browser:
                    self.browser.quit()
        except Exception:
            pass

        self.browser = None
        self.page = None
        self.context = None
        self.playwright = None

    def get_stats(self) -> Dict:
        """Get browser agent statistics."""
        return {
            "available": BROWSER_AVAILABLE,
            "backend": BROWSER_TYPE,
            "is_running": self.browser is not None,
            "history_count": len(self.history),
            "current_url": self.get_url(),
            "current_title": self.get_title(),
            "screenshots_dir": str(self.screenshots_dir),
        }

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def run_browser_command(agent: BrowserAgent, command: str) -> str:
    """
    Run a natural language browser command.

    Args:
        agent: BrowserAgent instance
        command: Natural language command

    Returns:
        Result string
    """
    command = command.lower().strip()

    if command.startswith("go to ") or command.startswith("open "):
        url = command.split(" ", 2)[-1]
        result = agent.goto(url)
        if result.get("success"):
            return f"Opened: {result['title']}"
        return f"Error: {result.get('error')}"

    elif command.startswith("search "):
        query = command[7:]
        results = agent.search_google(query)
        if results:
            output = f"Found {len(results)} results:\n"
            for r in results[:5]:
                output += f"- {r['text'][:50]} ({r['href'][:50]})\n"
            return output
        return "No results found"

    elif command in ("read", "content"):
        return agent.get_content(max_length=1000)

    elif command.startswith("click "):
        target = command[6:]
        result = agent.click(target)
        if result.get("success"):
            return f"Clicked: {target}"
        return f"Error: {result.get('error')}"

    elif command == "screenshot":
        path = agent.screenshot()
        return f"Screenshot saved: {path}"

    elif command == "links":
        links = agent.get_links()
        return "\n".join([f"- {lnk['text'][:30]} ({lnk['href'][:50]})" for lnk in links[:10]])

    elif command in ("scroll", "scroll down"):
        agent.scroll("down")
        return "Scrolled down"

    elif command == "scroll up":
        agent.scroll("up")
        return "Scrolled up"

    elif command == "close":
        agent.close()
        return "Browser closed"

    else:
        return "Commands: go to <url>, search <query>, read, click <element>, screenshot, links, scroll, close"


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("BROWSER AGENT TEST")
    print("=" * 60)

    if not BROWSER_AVAILABLE:
        print("\nBrowser automation not available!")
        print("Install one of:")
        print("  pip install playwright && playwright install chromium")
        print("  pip install selenium webdriver-manager")
    else:
        print(f"\nUsing: {BROWSER_TYPE}")

        with BrowserAgent(headless=True) as agent:
            print(f"\nStats: {agent.get_stats()}")

            # Test Google search
            print("\nSearching Google for 'Python programming'...")
            results = agent.search_google("Python programming")
            for r in results[:3]:
                print(f"  - {r['text'][:40]}")

            # Test Wikipedia
            print("\nGoing to Wikipedia...")
            result = agent.goto("https://en.wikipedia.org/wiki/Artificial_intelligence")
            print(f"  Title: {result.get('title')}")

            # Read content
            print("\nReading content...")
            content = agent.get_content(max_length=200)
            print(f"  {content}...")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
