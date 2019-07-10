alert("请输入分辨率为32x32的图像");
function show_orign(source, target) {

    var img = document.getElementById(source).files[0];
    // var url = window.URL.createObjectURL(img);
    // var tar_img = document.getElementById(target);
    // tar_img.src = url;
    // tar_img.width =256;
    // tar_img.height = 256;

    var data = new FormData();
    data.append('file', img);


    $.ajax({
        url: '/receive',
        type: 'POST',
        dataType: 'JSON',
        data: data,

        processData: false,//用于对data参数进行序列化处理 这里必须false
        contentType: false,
        success:function (res) {

            $('#orign_id').attr("width", 256);
            $('#orign_id').attr("height", 256);
            // $('#orign_id').attr("src", res['filename1']+"?Math.random()");
            var org = document.getElementById('orign_id');
            org.src = res['filename1']+"?t="+Math.random();

            $('#res1_id').attr("width", 256);
            $('#res1_id').attr("height", 256);
            var res_1 = document.getElementById('res1_id');
            res_1.src = res['filename2']+"?t="+Math.random();

            $('#res2_id').attr("width", 256);
            $('#res2_id').attr("height", 256);
            var res_2 = document.getElementById('res2_id');
            res_2.src = res['filename3']+"?t="+Math.random();


        },
        error: function (e) {
           alert(e);
        }
    });



}