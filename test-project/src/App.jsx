import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import { useState } from 'react';
import HomePage from './HomePage';
import CartPage from './CartPage';
import './App.css';

function App() {
  const [cartCount, setCartCount] = useState(0);

  const addToCart = () => {
    setCartCount(prev => prev + 1);
  };

  const clearCart = () => {
    setCartCount(0);
  };

  return (
    <Router>
      <div className="app">
        <nav className="navbar">
          <Link to="/" className="nav-brand">
            🛍️ ShopEasy
          </Link>
          <Link to="/cart" className="cart-link" id="cart-link">
            🛒 Cart
            {cartCount > 0 && <span className="cart-badge" id="cart-badge">{cartCount}</span>}
          </Link>
        </nav>
        
        <main className="main-content">
          <Routes>
            <Route path="/" element={<HomePage addToCart={addToCart} />} />
            <Route path="/cart" element={<CartPage cartCount={cartCount} clearCart={clearCart} />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
