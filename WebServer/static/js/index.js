$(function() {
    $("#classification-input-files").change((event) => {
        let input = event.target;
        if(input.files) {
            $("#classification-img-preview").html("");
            for (let i = 0; i < input.files.length; i++) {
                let reader = new FileReader();
                reader.onload = function() {
                    console.log('onload function called')
                    $($.parseHTML('<img>')).attr('src', this.result).attr('width', '100px').appendTo("#classification-img-preview");
                }
                reader.readAsDataURL(input.files[i]);
            }
        }
    });
});