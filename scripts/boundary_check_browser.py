"""Boundary check: verify Playwright can drive system Chrome and reach x.com.

Read-only: navigates, waits, screenshots. No clicks, no input.
"""
import json
import sys
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent.parent / "data" / "boundary"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> int:
    profile = Path(__file__).resolve().parent.parent / "data" / "browser-profile"
    profile.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch_persistent_context(
            str(profile),
            channel="chrome",
            headless=False,
            viewport={"width": 1440, "height": 900},
            args=["--disable-blink-features=AutomationControlled"],
        )
            page = browser.pages[0] if browser.pages else browser.new_page()
            page.goto("https://x.com", wait_until="domcontentloaded", timeout=45000)
            page.wait_for_timeout(6000)

            title = page.title()
            url = page.url
            # Login-state probes (no interaction, just DOM reads)
            has_login_form = page.locator("input[autocomplete='username']").count() > 0
            has_sidebar_nav = page.locator("nav[role='navigation']").count() > 0
            has_compose_button = page.locator(
                "a[data-testid='SideNav_NewTweet_Button'], button[data-testid='SideNav_NewTweet_Button']"
            ).count() > 0
            n_articles = page.locator("article[data-testid='tweet']").count()

            page.screenshot(path=str(OUT / "x_landing.png"))

            result = {
                "title": title,
                "url": url,
                "viewport": "1440x900",
                "login_form_present": has_login_form,
                "sidebar_nav_present": has_sidebar_nav,
                "compose_button_present": has_compose_button,
                "tweet_articles_visible": n_articles,
                "conclusion": "LOGGED_OUT" if has_login_form else "LOGGED_IN_OR_PARTIAL",
            }
            (OUT / "x_landing.json").write_text(json.dumps(result, indent=2))
            print(json.dumps(result, indent=2))
            browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
