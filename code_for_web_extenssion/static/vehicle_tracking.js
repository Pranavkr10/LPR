function searchVehicle() {
    document.getElementById('searchForm').addEventListener('submit', function(e){
        const last4Digit = document.getElementById('last4Digits').value;
        const vehicleColour = document.getElementById('vehicle-colour').value;

    if(!last4Digit && !vehicleColour){
        alert("Please enter at least one search criteria");
        e.preventDefault();
    }
});}