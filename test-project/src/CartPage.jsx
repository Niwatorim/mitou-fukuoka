import { Link } from 'react-router-dom';

function CartPage({ cartCount, clearCart }) {
    const itemPrice = 29.99;
    const totalPrice = (cartCount * itemPrice).toFixed(2);

    return (
        <div className="cart-page">
            <h1 className="page-title">Your Shopping Cart</h1>

            {cartCount === 0 ? (
                <div className="empty-cart">
                    <div className="empty-cart-icon">🛒</div>
                    <p>Your cart is empty</p>
                    <Link to="/" className="continue-shopping-btn">
                        Continue Shopping
                    </Link>
                </div>
            ) : (
                <div className="cart-content">
                    <div className="cart-items">
                        <div className="cart-item">
                            <div className="item-image">📦</div>
                            <div className="item-details">
                                <h3>Premium Product</h3>
                                <p className="item-price">${itemPrice}</p>
                            </div>
                            <div className="item-quantity">
                                <span>Quantity: </span>
                                <span id="cart-quantity" className="quantity-value">{cartCount}</span>
                            </div>
                        </div>
                    </div>

                    <div className="cart-summary">
                        <div className="summary-row">
                            <span>Items:</span>
                            <span id="total-items">{cartCount}</span>
                        </div>
                        <div className="summary-row total">
                            <span>Total:</span>
                            <span id="total-price">${totalPrice}</span>
                        </div>
                        <button
                            className="checkout-btn"
                            id="checkout-btn"
                            onClick={() => alert('Checkout functionality coming soon!')}
                        >
                            Proceed to Checkout
                        </button>
                        <button
                            className="clear-cart-btn"
                            id="clear-cart-btn"
                            onClick={clearCart}
                        >
                            Clear Cart
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
}

export default CartPage;
