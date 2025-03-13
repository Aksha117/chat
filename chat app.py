async function fetchStockData(symbol) {
    const apiUrl = `https://your-api.onrender.com/predict?symbol=${symbol}`;

    // Show loading animation
    document.getElementById("stock-data").innerHTML = `<p class="loading">⏳ Fetching stock data...</p>`;

    try {
        const response = await fetch(apiUrl);
        const data = await response.json();

        if (data.error) {
            document.getElementById("stock-data").innerHTML = `<p class="error">⚠️ Stock data not found!</p>`;
            return;
        }

        // Display Stock Data with Animation
        const stockDataDiv = document.getElementById("stock-data");
        stockDataDiv.innerHTML = `
            <div class="stock-box">
                <strong>${data.symbol}</strong><br>
                🏷️ Last Price: ₹${data.lastPrice} <br>
                📊 LSTM Predicted Price: ₹${data.predictedPrice.toFixed(2)} <br>
                ⚡ XGBoost Predicted Price: ₹${data.xgbPrediction.toFixed(2)} <br>
                📊 Trading Volume: ${data.volume} <br>
            </div>
        `;
        stockDataDiv.style.animation = "slideIn 1s forwards";
    } catch (error) {
        console.error("Error fetching stock data:", error);
        document.getElementById("stock-data").innerHTML = `<p class="error">⚠️ Error fetching stock data.</p>`;
    }
}

// Event Listener for Button Click
document.getElementById("predict-btn").addEventListener("click", function() {
    const symbol = document.getElementById("stock-symbol").value.toUpperCase().trim();
    if (symbol) {
        fetchStockData(symbol);
    } else {
        document.getElementById("stock-data").innerHTML = `<p class="error">⚠️ Please enter a stock symbol!</p>`;
    }
});
