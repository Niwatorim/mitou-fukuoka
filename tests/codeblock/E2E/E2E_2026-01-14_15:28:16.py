from playwright.sync_api import Page, expect
import re

def test_counter_increment(page: Page):
    # Navigate to the application
    page.goto("http://localhost:5173/")
    expect(page).to_have_url("http://localhost:5173/")
    expect(page).to_have_title(re.compile(r"Vite \+ React")) # Assuming a default Vite React title

    # Find the button with initial text "count is 0" and click it
    counter_button = page.get_by_role("button", name="count is 0")
    expect(counter_button).to_be_visible()
    counter_button.click()

    # Assert that the button's text has updated to "count is 1"
    expect(page.get_by_role("button", name="count is 1")).to_be_visible()
    expect(page.get_by_role("button")).to_have_text("count is 1")