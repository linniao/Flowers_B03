
function show_orign(source, target) {
    var img = document.getElementById(source).files[0];
    var url = window.URL.createObjectURL(img);
    var tar_img = document.getElementById(target);
    tar_img.src = url;
    tar_img.width =256;
    tar_img.height = 256;

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
            $('#res1_id').attr("width", 256);
            $('#res1_id').attr("height", 256);
            var res_1 = document.getElementById('res1_id');
            res_1.src = res['filename1']
        },
        error: function (e) {
           alert(e);
        }
    })
}