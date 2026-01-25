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
                import pandas as pd # Assuming pandas is available for pd.notna

                # Navigate to the registration page
                await page.goto("http://localhost:5173/")
                await expect(page).to_have_url("http://localhost:5173/")

                # Fill out the registration form
                username_value = str(row['Username']) if pd.notna(row['Username']) else "" #WHAT IS THIS
                email_value = str(row['Email Address']) if pd.notna(row['Email Address']) else ""
                password_value = str(row['Password']) if pd.notna(row['Password']) else ""
                confirm_password_value = str(row['Confirm Password']) if pd.notna(row['Confirm Password']) else ""
                country_value = str(row['Country']) if pd.notna(row['Country']) else ""
                terms_accepted_value = str(row['I accept the terms and conditions']).lower() == 'true'

                await page.locator("input#username").fill(username_value)
                await page.locator("input#email").fill(email_value)
                await page.locator("input#password").fill(password_value)
                await page.locator("input#confirmPassword").fill(confirm_password_value)
                await page.locator("select#country").select_option(country_value)

                if terms_accepted_value:
                    await page.locator("input#termsAccepted").check()
                else:
                    await page.locator("input#termsAccepted").uncheck()

                # Click the "Create Account" button
                await page.locator('button[ref="e37"]').click()
                await page.wait_for_load_state('networkidle')

                # Assert the success message
                result_element = page.locator('[ref="e38"]')
                await expect(result_element).to_be_visible()
                actual_result_text = await result_element.text_content()
                expected_message = "Registration successful!"
                assert expected_message.lower() in str(actual_result_text).lower(), \
                    f"Expected '{{expected_message}}' to be in '{{actual_result_text}}'"
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
