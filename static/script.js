$(document).ready(function () {
    $("#churn-form").on("submit", function (e) {
        e.preventDefault();

        // Disable button + show loading text
        $("#predict-btn").prop("disabled", true).text("Predicting...");

        // Build payload from form
        const payload = {
            CreditScore: parseInt($("#CreditScore").val()),
            Geography: $("#Geography").val(),
            Gender: $("#Gender").val(),
            Age: parseInt($("#Age").val()),
            Tenure: parseInt($("#Tenure").val()),
            Balance: parseFloat($("#Balance").val()),
            NumOfProducts: parseInt($("#NumOfProducts").val()),
            HasCrCard: parseInt($("#HasCrCard").val()),
            IsActiveMember: parseInt($("#IsActiveMember").val()),
            EstimatedSalary: parseFloat($("#EstimatedSalary").val())
        };

        // AJAX POST to FastAPI backend
        $.ajax({
            url: "/predict",
            method: "POST",
            contentType: "application/json",
            data: JSON.stringify(payload),
            success: function (response) {
                const pred = response.churn_prediction;
                const proba = response.churn_probability;

                const percent = (proba * 100).toFixed(2);

                let message = "";
                if (pred === 1) {
                    message = `⚠️ This customer is LIKELY to CHURN.<br>
                               Churn probability: <strong>${percent}%</strong>`;
                } else {
                    message = `✅ This customer is UNLIKELY to churn.<br>
                               Churn probability: <strong>${percent}%</strong>`;
                }

                $("#result").html(message);
            },
            error: function (xhr, status, error) {
                console.error(error);
                $("#result").html("❌ Error making prediction. Check console/logs.");
            },
            complete: function () {
                // Re-enable button
                $("#predict-btn").prop("disabled", false).text("Predict Churn");
            }
        });
    });
});
