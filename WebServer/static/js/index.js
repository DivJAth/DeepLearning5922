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

$("#fm-classification-input-files").change((event) => {
    let input = event.target;
    if(input.files) {
        $("#fm-classification-img-preview").html("");
        for (let i = 0; i < input.files.length; i++) {
            let reader = new FileReader();
            reader.onload = function() {
                $($.parseHTML("<img>")).attr("src", this.result).attr("width", "100px").appendTo("#fm-classification-img-preview");
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
});

$("#fm-classification-form").submit((event) => {
    const formData = new FormData(event.target);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/ml/fm-classification", true);
    xhr.onreadystatechange = () => {
        if(xhr.readyState == 4) {
            console.log(xhr.responseText);
        }
    };
    xhr.send(formData);
    event.preventDefault();
<<<<<<< HEAD
<<<<<<< HEAD
});

$("#object-detection-form").submit((event) => {
    const formData = new FormData(event.target);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/ml/objectdetection", true);
    xhr.onreadystatechange = () => {
        if(xhr.readyState == 4) {
            const img_element = document.getElementById("object-detection-results");
            img_element.src = xhr.responseText;
        }
    };
    xhr.send(formData);
    event.preventDefault();
});

$("#progressive-gan-generation-button").click((event) => {
    const number = document.getElementById("random_number").value;
    const xhr = new XMLHttpRequest();
    const url = "/ml/progressive-gan-generation?number=" + number;
    console.log("Making request to " + url);
    xhr.open("GET", url, true);
    xhr.onreadystatechange = () => {
        if(xhr.readyState == 4) {
            const img_element = document.getElementById("progressive-gan-generation-results");
            img_element.src = xhr.responseText;
        }
    };
    xhr.send();
=======
>>>>>>> refs/remotes/origin/master
=======
>>>>>>> refs/remotes/origin/master
});