$("#classification-input-files").change((event) => {
    let input = event.target;
    if(input.files) {
        $("#classification-img-preview").html("");
        for (let i = 0; i < input.files.length; i++) {
            let reader = new FileReader();
            reader.onload = function() {
                $($.parseHTML("<img>")).attr("src", this.result).attr("width", "100px").appendTo("#classification-img-preview");
            }
            reader.readAsDataURL(input.files[i]);
        }
    }
});

$("#classification-form").submit((event) => {
    const formData = new FormData(event.target);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/ml/classification", true);
    xhr.onreadystatechange = () => {
        if(xhr.readyState == 4) {
            console.log(xhr.responseText);
        }
    };
    xhr.send(formData);
    event.preventDefault();
})