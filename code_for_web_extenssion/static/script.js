function editPlate(spandId, inputId, btn){
    var span = document.getElementById(spandId);
    var input = document.getElementById(inputId);

    if(input.style.display === "none"){
        input.style.display = "inline-block";
        input.value = span.innerText;
        span.style.display = "none";
        btn.innerText = 'Save';
    }else{
        span.innerText = input.value;
        span.style.display = "inline";
        input.style.display = "none";
        btn.innerText = "Edit"
    }
}

//show vehicle details
function showDetails(detailsHTML){
    const placeholder = document.getElementById(' detailsPlaceholder');
    const detailsDiv = document.getElementById(' vehicleDetails');
    //to hide placeholder
    placeholder.style.display='none';
    //Set the details for html
    detailsDiv.innerHTML = detailsHTML;
    //applying style on the table
    const table = detailsDiv.querySelector('table');
    if(table){
        table.className = 'detail-table';
        table.style.width = '100%';
        table.style.marginTop = '0';
    }
    detailsDiv.style.display = 'block';
}

//upload function
document.getElementById('browseBtn').addEventListener('click', function(){
    document.getElementById('fileInput').click();
});

 document.getElementById('fileInput').addEventListener('change', function(e){ 
    if(e.target.files.length > 0){
        const fileName = e.target.files[0].name;
        const uploadText = document.querySelector('.upload-text');
        uploadText.innerHTML=`<h3>File selected</h3><p>${fileName}</p>`;
    }
});
//alert for stolen and wanted vehicle
document.addEventListener("DOMContentLoaded", function(){
    const rows = document.querySelectorAll("tbody tr[data-category]");
    rows.forEach(row=>{
        const category = row.getAttribute("data-category");
        const plate = row.getAttribute("data-number");
        if(category === "stolen" || category === "wanted"){
            alert(`ALERT: Vehicle "${plate}" is marked as ${category.toUpperCase()}!`);
        }
    })
});