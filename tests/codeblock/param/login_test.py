import pandas as pd
import os
import asyncio
from playwright.async_api import async_playwright, Page, expect

csv_path = r"/Users/niwatorimostiqo/Desktop/Coding/Mitou-Fukuoka/tests/csv_s/login_test.csv"
df = pd.read_csv(csv_path)

print(f"Running {len(df)} test cases from CSV")

async def main():
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=False)

        for index, row in df.iterrows():
            print(f"\n=== Test Case {index + 1}/{len(df)} ===")
            print(f"Input values: {dict(row)}")
            context = await browser.new_context()
            page = await context.new_page()        
            try:
                await page.goto("http://localhost:5173/")
                await page.wait_for_load_state("domcontentloaded")

                # Fill email address
                email_value = row["email"]
                await page.get_by_label("Email Address").fill(email_value)

                # Fill password
                password_value = row["password"]
                await page.get_by_label("Password").fill(password_value)

                # Click the Log In button
                await page.get_by_role("button", name="Log In").click()
                await page.wait_for_load_state("networkidle") # Wait for network activity to cease after login attempt

                # Initialize actual response variables
                actual_response_email = ""
                actual_response_password = ""

                # Get expected response values from the CSV
                expected_response_email = row["expected_response_email"]
                expected_response_password = row["expected_response_password"]

                # Extract actual response for email
                # If an expected email message is provided, check if it's visible on the page.
                # If found, set actual_response_email to the expected message. Otherwise, it remains empty.
                if expected_response_email:
                    email_error_locator = page.locator(f"text='{expected_response_email}'")
                    if await email_error_locator.is_visible():
                        actual_response_email = expected_response_email

                # Extract actual response for password
                # If an expected password message is provided, check if it's visible on the page.
                # If found, set actual_response_password to the expected message. Otherwise, it remains empty.
                if expected_response_password:
                    password_error_locator = page.locator(f"text='{expected_response_password}'")
                    if await password_error_locator.is_visible():
                        actual_response_password = expected_response_password

                # Assertions for email response
                assert actual_response_email == expected_response_email, \
                    f"Email response mismatch. Expected: '{expected_response_email}', Got: '{actual_response_email}'"

                # Assertions for password response
                assert actual_response_password == expected_response_password, \
                    f"Password response mismatch. Expected: '{expected_response_password}', Got: '{actual_response_password}'"
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