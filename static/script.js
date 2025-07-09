document.getElementById('prediction-form').addEventListener('submit', async function (e) {
    e.preventDefault();

    const cryptoPair = document.getElementById('crypto-pair').value;

    const response = await fetch(`/predict?crypto_pair=${cryptoPair}`);
    const data = await response.json();

    const resultContainer = document.getElementById('result-container');
    const resultElement = document.getElementById('result');
    const canvasElement = document.getElementById('prediction-graph');
    const tableBody = document.getElementById('table-body');

    if (data.error) {
        resultElement.innerHTML = `<strong>Error:</strong> ${data.error}`;
        resultContainer.style.display = 'block';
        return;
    }

    resultElement.innerHTML = `
        <strong>Prediction for ${cryptoPair}:</strong> $${data.predicted_price.toFixed(2)}
        <br>MSE: ${data.mse.toFixed(4)}
        <br>RMSE: ${data.rmse.toFixed(4)}
        <br>Accuracy: ${data.accuracy.toFixed(2)}%
    `;
    
    resultContainer.style.display = 'block';

    // Display data in the table
    tableBody.innerHTML = '';
    data.data_points.forEach(point => {
        const row = `<tr>
            <td>${point.date}</td>
            <td>${point.open}</td>
            <td>${point.high}</td>
            <td>${point.low}</td>
            <td>${point.close}</td>
        </tr>`;
        tableBody.innerHTML += row;
    });

    // Draw the graph
    const ctx = canvasElement.getContext('2d');
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.dates,
            datasets: [{
                label: 'Original Price',
                data: data.actual_prices,
                borderColor: 'green',
                backgroundColor: 'rgba(0, 255, 0, 0.2)',
                borderWidth: 2
            }, {
                label: 'Predicted Price',
                data: data.predicted_prices,
                borderColor: 'red',
                backgroundColor: 'rgba(255, 0, 0, 0.2)',
                borderWidth: 2
            }]
        }
    });
});
