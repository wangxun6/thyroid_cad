function inputChange()
{
    var file = document.getElementById("btn_file").files;
    console.log(file);
    var reader = new FileReader();
    reader.readAsDataURL(file[0]);
    document.getElementById("fname").innerHTML=file[0];
    reader.onload = function ()
    {
        var image = document.getElementById("picturebox");
        image.src = reader.result;
    };
}
function F_Open_dialog()
{
	document.getElementById("btn_file").click();
}


