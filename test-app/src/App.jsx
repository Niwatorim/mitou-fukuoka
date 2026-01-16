import { useState } from 'react'
import './index.css'

function LoginForm() {
    const [email, setEmail] = useState('')
    const [password, setPassword] = useState('')
    const [result, setResult] = useState(null)

    const handleSubmit = (e) => {
        e.preventDefault()

        // Simple validation logic for testing
        let status = 'success'
        let message = 'Login successful!'

        if (!email || !email.includes('@')) {
            status = 'error'
            message = 'Invalid email address'
        } else if (!password || password.length < 6) {
            status = 'error'
            message = 'Password must be at least 6 characters'
        } else if (email === 'admin@test.com' && password === 'admin123') {
            status = 'success'
            message = 'Welcome back, Admin!'
        } else if (email === 'test@example.com' && password === 'test123') {
            status = 'success'
            message = 'Welcome back, User!'
        } else {
            status = 'error'
            message = 'Invalid credentials'
        }

        setResult({ status, message })

        // Add test result to DOM for easy verification
        const resultDiv = document.getElementById('login-result')
        if (resultDiv) {
            resultDiv.textContent = status
            resultDiv.setAttribute('data-email-status', status)
            resultDiv.setAttribute('data-password-status', status)
        }
    }

    return (
        <div className="form-card">
            <h2>Login</h2>
            <p>Enter your credentials to continue</p>
            <form onSubmit={handleSubmit}>
                <div className="form-group">
                    <label htmlFor="login-email">Email Address</label>
                    <input
                        type="email"
                        id="login-email"
                        name="email"
                        placeholder="your@email.com"
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                    />
                </div>
                <div className="form-group">
                    <label htmlFor="login-password">Password</label>
                    <input
                        type="password"
                        id="login-password"
                        name="password"
                        placeholder="Enter your password"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                    />
                </div>
                <button type="submit" className="submit-btn">
                    Log In
                </button>
            </form>
            {result && (
                <div className={`result-message ${result.status}`} id="login-result">
                    {result.message}
                </div>
            )}
        </div>
    )
}

function RegistrationForm() {
    const [formData, setFormData] = useState({
        username: '',
        email: '',
        password: '',
        confirmPassword: '',
        country: '',
        termsAccepted: false
    })
    const [result, setResult] = useState(null)

    const handleChange = (e) => {
        const { name, value, type, checked } = e.target
        setFormData(prev => ({
            ...prev,
            [name]: type === 'checkbox' ? checked : value
        }))
    }

    const handleSubmit = (e) => {
        e.preventDefault()

        let status = 'success'
        let message = 'Registration successful!'

        // Validation logic
        if (!formData.username || formData.username.length < 3) {
            status = 'error'
            message = 'Username must be at least 3 characters'
        } else if (!formData.email || !formData.email.includes('@')) {
            status = 'error'
            message = 'Invalid email address'
        } else if (!formData.password || formData.password.length < 8) {
            status = 'error'
            message = 'Password must be at least 8 characters'
        } else if (formData.password !== formData.confirmPassword) {
            status = 'error'
            message = 'Passwords do not match'
        } else if (!formData.country) {
            status = 'warning'
            message = 'Please select a country'
        } else if (!formData.termsAccepted) {
            status = 'error'
            message = 'You must accept the terms and conditions'
        }

        setResult({ status, message })

        // Add test results to DOM
        const resultDiv = document.getElementById('registration-result')
        if (resultDiv) {
            resultDiv.textContent = status
            resultDiv.setAttribute('data-username-status', status)
            resultDiv.setAttribute('data-email-status', status)
            resultDiv.setAttribute('data-password-status', status)
            resultDiv.setAttribute('data-country-status', status)
        }
    }

    return (
        <div className="form-card">
            <h2>Register</h2>
            <p>Create a new account</p>
            <form onSubmit={handleSubmit}>
                <div className="form-group">
                    <label htmlFor="reg-username">Username</label>
                    <input
                        type="text"
                        id="reg-username"
                        name="username"
                        placeholder="Choose a username"
                        value={formData.username}
                        onChange={handleChange}
                    />
                </div>
                <div className="form-group">
                    <label htmlFor="reg-email">Email Address</label>
                    <input
                        type="email"
                        id="reg-email"
                        name="email"
                        placeholder="your@email.com"
                        value={formData.email}
                        onChange={handleChange}
                    />
                </div>
                <div className="form-group">
                    <label htmlFor="reg-password">Password</label>
                    <input
                        type="password"
                        id="reg-password"
                        name="password"
                        placeholder="Create a password"
                        value={formData.password}
                        onChange={handleChange}
                    />
                </div>
                <div className="form-group">
                    <label htmlFor="reg-confirm-password">Confirm Password</label>
                    <input
                        type="password"
                        id="reg-confirm-password"
                        name="confirmPassword"
                        placeholder="Confirm your password"
                        value={formData.confirmPassword}
                        onChange={handleChange}
                    />
                </div>
                <div className="form-group">
                    <label htmlFor="reg-country">Country</label>
                    <select
                        id="reg-country"
                        name="country"
                        value={formData.country}
                        onChange={handleChange}
                    >
                        <option value="">Select a country</option>
                        <option value="USA">United States</option>
                        <option value="UK">United Kingdom</option>
                        <option value="Japan">Japan</option>
                        <option value="Canada">Canada</option>
                        <option value="Australia">Australia</option>
                    </select>
                </div>
                <div className="checkbox-group">
                    <input
                        type="checkbox"
                        id="reg-terms"
                        name="termsAccepted"
                        checked={formData.termsAccepted}
                        onChange={handleChange}
                    />
                    <label htmlFor="reg-terms">I accept the terms and conditions</label>
                </div>
                <button type="submit" className="submit-btn">
                    Create Account
                </button>
            </form>
            {result && (
                <div className={`result-message ${result.status}`} id="registration-result">
                    {result.message}
                </div>
            )}
        </div>
    )
}

function App() {
    return (
        <div className="app">
            <LoginForm />
            <RegistrationForm />
        </div>
    )
}

export default App
