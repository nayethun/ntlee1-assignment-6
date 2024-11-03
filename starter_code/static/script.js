document.addEventListener("DOMContentLoaded", () => {
    const generateButton = document.querySelector("#generate-button");
    const resultSection = document.querySelector("#results");
    const loadingSpinner = document.querySelector("#loading-spinner");

    // Check if the elements exist before manipulating them
    if (generateButton && loadingSpinner) {
        // Add a loading spinner while generating plots
        generateButton.addEventListener("click", (e) => {
            e.preventDefault();  // Prevent the form from submitting immediately

            // Hide the results section if it exists
            if (resultSection) {
                resultSection.style.display = "none";
            }

            // Show loading spinner
            loadingSpinner.style.display = "block";

            // Submit form after a slight delay to show spinner
            setTimeout(() => {
                e.target.closest("form").submit();
            }, 500);
        });
    }

    // Once the page is loaded, hide the loading spinner and show the results section
    if (loadingSpinner && resultSection) {
        loadingSpinner.style.display = "none";
        resultSection.style.display = "block";
    }
});
