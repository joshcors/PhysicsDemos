var socket = io();

socket.pingInterval = 10000;
socket.pingTimeout = 10000;

var $real_lower = $("#real_lower");
var $real_upper = $("#real_upper");
var $imag_lower = $("#imag_lower");
var $imag_upper = $("#imag_upper");
var $x_res = $("#x_res");
var $y_res = $("#y_res");
var $n_iter = $("#n_iter");

var $render = $("#render");

var $rendered_img = $("#rendered_img").hide();
var $spinner_container = $("#spinner-container").hide();

$render.on("click", function() {
    socket.emit("render_mandelbrot", {
        real_lower: $real_lower.val(),
        real_upper: $real_upper.val(),
        imag_lower: $imag_lower.val(),
        imag_upper: $imag_upper.val(),
        x_res: $x_res.val(),
        y_res: $y_res.val(),
        n_iter: $n_iter.val()
    });

    $spinner_container.show();
    $rendered_img.hide();
});

socket.on("rendered_mandelbrot", function(data) {
    $rendered_img.attr("src", data.image);
    $rendered_img.show();
    $spinner_container.hide();
});