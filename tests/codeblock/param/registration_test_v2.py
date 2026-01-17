import pandas as pd
import os
import asyncio
from playwright.async_api import async_playwright, Page, expect
#OLD

csv_path = r"/Users/niwatorimostiqo/Desktop/Coding/Mitou-Fukuoka/tests/csv_s/registration_test_v2.csv"
df = pd.read_csv(csv_path)

print(f"Running {len(df)} test cases from CSV")

async def main():
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=False)

        for index, row in df.iterrows():
            print(f"\\n=== Test Case {index + 1}/{len(df)} ===")
            print(f"Input values: {dict(row)}")
            context = await browser.new_context()
            page = await context.new_page()        
            try:
				await page.goto("http://localhost:5173/")
				await page.wait_for_load_state("domcontentloaded")

				# Fill input fields
				username_value = row["username"]
				email_value = row["email"]
				password_value = row["password"]
				country_value = row["country"]

				# Assuming specific IDs for input fields based on best practices
				await page.fill("#username-input", username_value)
				await page.fill("#email-input", email_value)
				await page.fill("#password-input", password_value)
				# Assuming country is a select dropdown
				await page.select_option("#country-select", value=country_value)

				# Click the submit button
				# Assuming a submit button with a specific ID or text
				await page.click("#register-button") # Or page.get_by_role("button", name="Register").click()
				await page.wait_for_load_state("networkidle")

				# Extract actual results from the page
				# Assuming the response values are displayed in elements with specific IDs
				actual_response_username = await page.locator("#response-username").text_content()
				actual_response_email = await page.locator("#response-email").text_content()
				actual_response_password = await page.locator("#response-password").text_content()
				actual_response_country = await page.locator("#response-country").text_content()

				# Compare actual vs. expected results
				expected_response_username = row["expected_response_username"]
				expected_response_email = row["expected_response_email"]
				expected_response_password = row["expected_response_password"]
				expected_response_country = row["expected_response_country"]

				assert actual_response_username == expected_response_username, \
					f"Username mismatch: Expected '{expected_response_username}', Got '{actual_response_username}'"
				assert actual_response_email == expected_response_email, \
					f"Email mismatch: Expected '{expected_response_email}', Got '{actual_response_email}'"
				assert actual_response_password == expected_response_password, \
					f"Password mismatch: Expected '{expected_response_password}', Got '{actual_response_password}'"
				assert actual_response_country == expected_response_country, \
					f"Country mismatch: Expected '{expected_response_country}', Got '{actual_response_country}'"
                except Exception as e:
                    print(f"Test case {index + 1} FAILED: {e}")
                else:
                    print(f"Test case {index + 1} PASSED")
                finally:
                    context.close()
        browser.close()

if __name__ == "__main__":
    asyncio.run(main())
    print("\\nAll tests completed!")
