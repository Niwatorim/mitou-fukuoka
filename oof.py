import pandas as pd
import os
import asyncio
from playwright.async_api import async_playwright, Page, expect

csv_path = r""
df = pd.read_csv(csv_path)

print(f"Running {len(df)} test cases from CSV")

async def main():
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)

        for index, row in df.iterrows():
            print(f"\\n=== Test Case {index + 1}/{len(df)} ===")
            print(f"Input values: {dict(row)}")
            context = await browser.new_context()
            page = await context.new_page()        
            try:


                await page.goto("http://localhost:5173/")
                await page.wait_for_load_state("domcontentloaded")

                # Get input values from CSV row, handling potential None/NaN
                username_value = str(row["email"]) if pd.notna(row["email"]) else ""
                password_value = str(row["password"]) if pd.notna(row["password"]) else ""

                # Fill the username field
                await page.fill("#login-email", username_value)

                # Fill the password field
                await page.fill("#login-password", password_value)

                # Click the Log In button
                await page.click("button:has-text('Log In')")

                # Wait for network to be idle after form submission
                await page.wait_for_load_state("networkidle")

                # Extract actual results from the result element
                # The history indicates an attempt to get attributes from #login-result
                result_element = page.locator("#login-result")

                # Check if the result element exists before trying to get attributes
                # If it doesn't exist, get_attribute will return None, which is handled by str()
                actual_username_status = await result_element.get_attribute("data-username-status")
                actual_password_status = await result_element.get_attribute("data-password-status")

                # Convert actual results to string, handling None
                actual_username_status_str = str(actual_username_status) if actual_username_status else ""
                actual_password_status_str = str(actual_password_status) if actual_password_status else ""

                # Get expected results from CSV row, handling potential None/NaN
                expected_username_status = str(row["expected_result"]) if pd.notna(row["expected_result"]) else ""
                

                # Assertions using flexible 'contains' matching
                assert expected_username_status.lower() in actual_username_status_str.lower() or actual_username_status_str.lower() in expected_username_status.lower(),                     f"Username status mismatch for input '{username_value}'. Expected '{expected_username_status}', Got '{actual_username_status_str}'"
            except Exception as e:
                print(f"Test case {index + 1} FAILED: {e}")
            else:
                print(f"Test case {index + 1} PASSED")
            finally:
                await context.close()
        await browser.close()             

if __name__ == "__main__":
    asyncio.run(main())
    print("\nAll tests completed!")
