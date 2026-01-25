import pandas as pd
import os
import asyncio
from playwright.async_api import async_playwright, Page, expect
async def main():
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=False,slow_mo=50)
        context = await browser.new_context()
        page = await context.new_page()        
        try:

                import re
                from playwright.async_api import Page, expect

                # Scenario 1: Add an item to the cart from the home page.
                await page.goto("http://localhost:5173/")
                await page.locator("#add-to-cart-btn").click()

                # Assert cart badge text after adding an item
                cart_badge_locator = page.locator("#cart-badge")
                actual_badge_text_add = await cart_badge_locator.text_content()
                expected_badge_text_add = "1"
                assert expected_badge_text_add.lower() in str(actual_badge_text_add).lower(), \
                    f"Expected '{expected_badge_text_add}' to be in '{actual_badge_text_add}' after adding item."
                await expect(cart_badge_locator).to_contain_text(expected_badge_text_add)

                # Scenario 2: Clear the cart from the cart page.
                await page.goto("http://localhost:5173/cart")
                await page.locator("#clear-cart-btn").click()

                # Assert cart badge text after clearing the cart
                # Playwright's expect will automatically wait for the element to update.
                actual_badge_text_clear = await cart_badge_locator.text_content()
                expected_badge_text_clear = "0"
                assert expected_badge_text_clear.lower() in str(actual_badge_text_clear).lower(), \
                    f"Expected '{expected_badge_text_clear}' to be in '{actual_badge_text_clear}' after clearing cart."
                await expect(cart_badge_locator).to_contain_text(expected_badge_text_clear)

                # Assert "Your cart is empty" message is visible on the page
                await expect(page.get_by_text("Your cart is empty")).to_be_visible()
        except Exception as e:
            print(f"Test case FAILED: {e}")
        else:
            print(f"Test case PASSED")
        finally:
            await context.close()
        await browser.close()             

if __name__ == "__main__":
    asyncio.run(main())
    print("\nAll tests completed!")
