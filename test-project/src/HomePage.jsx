function HomePage({ addToCart }) {
    return (
        <div className="home-page">
            <h1 className="page-title">Welcome to ShopEasy</h1>
            <p className="page-subtitle">Click the button to add items to your cart!</p>

            <div className="product-card">
                <div className="product-image">📦</div>
                <h2 className="product-title">Premium Product</h2>
                <p className="product-price">$29.99</p>
                <p className="product-description">
                    A fantastic item that you definitely need in your life.
                    High quality and amazing value!
                </p>
                <button
                    className="add-to-cart-btn"
                    id="add-to-cart-btn"
                    onClick={addToCart}
                >
                    Add to Cart
                </button>
            </div>
        </div>
    );
}

export default HomePage;
