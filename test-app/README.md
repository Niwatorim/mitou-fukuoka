# Parameter Testing Demo App

A simple React application designed to test the parameter testing feature of your Mitou-Fukuoka testing pipeline.

## Features

### 1. Login Form
- Email input field
- Password input field
- Validation logic
- Success/error messages

### 2. Registration Form
- Username input field
- Email input field
- Password input field
- Confirm password field
- Country dropdown
- Terms & conditions checkbox
- Comprehensive validation

## Setup

1. Install dependencies:
```bash
cd test-app
npm install
```

2. Run the development server:
```bash
npm run dev
```

The app will be available at `http://localhost:5173`

## Testing with Parameter Testing Pipeline

### Login Form Tests

Use the CSV file: `tests/csv_s/login_test.csv`

**Test scenarios included:**
- Valid credentials (`admin@test.com` / `admin123`)
- Valid credentials (`test@example.com` / `test123`)
- Invalid email format
- Short password
- Empty fields
- Wrong credentials

### Registration Form Tests

Use the CSV file: `tests/csv_s/registration_test.csv`

**Test scenarios included:**
- Valid registration
- Short username
- Invalid email
- Short password
- Missing country
- Empty username
- Various combinations of valid/invalid fields

## Form Validation Rules

### Login Form
- Email: Must contain `@`
- Password: Minimum 6 characters
- Valid accounts:
  - `admin@test.com` / `admin123`
  - `test@example.com` / `test123`

### Registration Form
- Username: Minimum 3 characters
- Email: Must contain `@`
- Password: Minimum 8 characters
- Confirm Password: Must match password
- Country: Must be selected
- Terms: Must be accepted

## Expected Results

Each form displays result messages with specific classes:
- `.result-message.success` - Validation passed
- `.result-message.error` - Validation failed
- `.result-message.warning` - Warning message

The result divs also have `data-*-status` attributes for programmatic verification.

## How to Use with Your Pipeline

1. Start this app: `npm run dev`
2. In your main test page, select "Parameter" test type
3. Upload one of the CSV files (`login_test.csv` or `registration_test.csv`)
4. Describe the test: 
   - "Test the login form" or
   - "Test the registration form"
5. Let the AI generate the parameter test script
6. Run the tests from the Parameter Testing page
7. View results for each CSV row

## File Structure

```
test-app/
├── src/
│   ├── App.jsx          # Main React components
│   ├── main.jsx         # React entry point
│   └── index.css        # Styling
├── index.html           # HTML template
├── package.json         # Dependencies
└── vite.config.js       # Vite configuration
```

## Test Data Location

CSV test files are located in:
- `../tests/csv_s/login_test.csv`
- `../tests/csv_s/registration_test.csv`
