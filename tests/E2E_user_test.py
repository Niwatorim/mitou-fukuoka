import re
from playwright.sync_api import Page, expect

def test_counter_interactions(page: Page):
    # Step 1: Navigate to the application
    page.goto("http://localhost:5173/")
    expect(page).to_have_url("http://localhost:5173/")

    # Locate the counter button using a regex to match its changing text
    counter_button = page.get_by_role("button", name=r"count is")

    # Assert initial state
    expect(counter_button).to_have_text("count is 0")

    # Step 2: Click the button to increment the count from 0 to 1
    counter_button.click()
    expect(counter_button).to_have_text("count is 1")

    # Step 3: Click the button again to increment the count from 1 to 2
    counter_button.click()
    expect(counter_button).to_have_text("count is 2")

    # Step 4: Navigate again, which typically resets the application state (e.g., counter back to 0)
    page.goto("http://localhost:5173/")
    expect(page).to_have_url("http://localhost:5173/")
    # Assert that the counter has reset to its initial state
    expect(counter_button).to_have_text("count is 0")

    # Step 5: Click the button to increment the count from 0 to 1 again
    counter_button.click()
    expect(counter_button).to_have_text("count is 1")

    # Step 6: Click the button again to increment the count from 1 to 2
    counter_button.click()
    expect(counter_button).to_have_text("count is 2")