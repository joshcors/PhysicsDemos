
import * as PIXI from './pixi.mjs';

var socket = io();

socket.pingInterval = 10000;
socket.pingTimeout = 10000;

const app = new PIXI.Application({
    width: 500,
    height: 500
});
document.body.appendChild(app.view);

app.stage.interactive = true;

var $pause_play = $("#pause-play");
var $draw_trail = $("#draw-trail");

var $n_pendula = $("#n_pendula");

var $d_theta_1 = $("#d_theta_1");
var $d_theta_2 = $("#d_theta_2");

var colors = [
    0xff0000,
    0xff7f00,
    0xffff00,
    0x00ff00,
    0x00ffff,
    0x0000ff,
    0x7f00ff,
    0xff00ff,
    0xff007f,
    0xff0000
];

function set_pendula() {
    var n_pendula = $n_pendula.val();
    var thetas_1 = get_thetas_1();
    var thetas_2 = get_thetas_2();

    for (var i = pendula.length - 1; i >= 0; i--) {
        pendula[i].destroy();
        pendula.pop();
    }

    for (var i = 0; i < n_pendula; i++) {
        var pendulum = new Pendulum(0, 0, 100, 100, i);
        pendulum.update(thetas_1[i], thetas_2[i]);
        pendulum.draw();
        pendula.push(pendulum);
    }
}

$n_pendula.on("input", function() {
    var n_pendula = $n_pendula.val();
    if (n_pendula > 10) {
        n_pendula = 10;
        $n_pendula.val(10);
    }
    
    set_pendula();
})

var keep_updating = false;

function play() {
    var theta_1 = [];
    var theta_2 = [];
    for (var i = 0; i < $n_pendula.val(); i++) {
        theta_1.push(pendula[i].theta_1);
        theta_2.push(pendula[i].theta_2);
    }
    socket.emit("play", {theta_1: theta_1, theta_2: theta_2});
    $pause_play.text("Stop");
    keep_updating = true;
}

function pause() {
    socket.emit("pause");
    $pause_play.text("Play");
    keep_updating = false;
}

$pause_play.on("click", function() {
    if ($pause_play.text() == "Play") {
        play();
    } else {
        pause();
    }
})

var $theta_1_slider = $("#theta_1_slider");
var $theta_1_value = $("#theta_1_value");
var $theta_2_slider = $("#theta_2_slider");
var $theta_2_value = $("#theta_2_value");

function get_thetas_1() {
    var thetas_1 = [$theta_1_slider.val() / 180 * Math.PI];
    for (var i = 1; i < $n_pendula.val(); i++) {
        thetas_1.push(thetas_1[i - 1] + $d_theta_1.val() / 180 * Math.PI);
    }
    return thetas_1;
}

function get_thetas_2() {
    var thetas_2 = [$theta_2_slider.val() / 180 * Math.PI];
    for (var i = 1; i < $n_pendula.val(); i++) {
        thetas_2.push(thetas_2[i - 1] + $d_theta_2.val() / 180 * Math.PI);
    }
    return thetas_2;
}


$theta_1_slider.on("input", function() {
    if ($pause_play.text() == "Stop") {
        pause();
    }
    set_pendula();
    $theta_1_value.text($theta_1_slider.val());
    socket.emit("update", {theta_1: get_thetas_1(), theta_2: get_thetas_2()});
})

$theta_2_slider.on("input", function() {
    if ($pause_play.text() == "Stop") {
        pause();
    }
    set_pendula();
    $theta_2_value.text($theta_2_slider.val());
    socket.emit("update", {theta_1: get_thetas_1(), theta_2: get_thetas_2()});
})

class Pendulum {
    constructor(theta_1, theta_2, length_1, length_2, index) {
        this.theta_1 = theta_1;
        this.theta_2 = theta_2;
        this.length_1 = length_1;
        this.length_2 = length_2;

        this.x1 = 0;
        this.y1 = 0;
        this.x2 = 0;
        this.y2 = 0;

        this.trail_x = [];
        this.trail_y = [];

        this.max_trail = 500;
        this.trail = new PIXI.Graphics();
        
        this.line = new PIXI.Graphics();
        this.color = colors[index];
    }

    get_cartesian() {

        var x1 = app.screen.width/2 + this.length_1 * Math.sin(this.theta_1);
        var y1 = (app.screen.height/2 + -this.length_1 * Math.cos(this.theta_1));
        var x2 = x1 + this.length_2 * Math.sin(this.theta_2);
        var y2 = (y1 - this.length_2 * Math.cos(this.theta_2));

        y1 = app.screen.height - y1;
        y2 = app.screen.height - y2;

        return [x1, y1, x2, y2];
    }

    update(theta_1, theta_2) {
        this.theta_1 = theta_1;
        this.theta_2 = theta_2;

        var values = this.get_cartesian();

        var x2 = values[2];
        var y2 = values[3];

        this.trail_x.push(x2);
        this.trail_y.push(y2);

        if (this.trail_x.length > this.max_trail) {
            this.trail_x.shift();
            this.trail_y.shift();
        }
    }

    draw() {

        var values = this.get_cartesian();

        this.x1 = values[0];
        this.y1 = values[1];
        this.x2 = values[2];
        this.y2 = values[3];

        app.stage.removeChild(this.line);
        this.line.clear();
        this.line.lineStyle(2, 0xFFFFFF);
        this.line.moveTo(app.screen.width / 2, app.screen.height / 2);
        this.line.lineTo(this.x1, this.y1);
        this.line.moveTo(this.x1, this.y1);
        this.line.lineTo(this.x2, this.y2);
        this.line.endFill();
        app.stage.addChild(this.line);

        if ($draw_trail.prop("checked")) {
            app.stage.removeChild(this.trail);
            this.trail.clear();
            this.trail.lineStyle(1, this.color);
            this.trail.moveTo(this.trail_x[0], this.trail_y[0]);
            for (var i = 1; i < this.trail_x.length; i++) {
                this.trail.lineTo(this.trail_x[i], this.trail_y[i]);
            }
            this.trail.endFill();
            app.stage.addChild(this.trail);
        }
        else {
            app.stage.removeChild(this.trail);
            this.trail_x = [];
            this.trail_y = [];
        }
    }

    destroy() {
        app.stage.removeChild(this.trail);
        app.stage.removeChild(this.line);
    }
}

var pendula = [new Pendulum(0, 0, 100, 100, 0)];

for (var i = 0; i < $n_pendula.val(); i++) {
    pendula[i].draw();
}

socket.on("update", function(data) {
    for (var i = 0; i < $n_pendula.val(); i++) {
        pendula[i].update(data.theta_1[i], data.theta_2[i]);
        pendula[i].draw();
    }
    $theta_1_slider.val(((data.theta_1[0] * 180 / Math.PI) % 360 + 360) % 360);
    $theta_2_slider.val(((data.theta_2[0] * 180 / Math.PI) % 360 + 360) % 360);
})



$(document).ready(function() {
    $("#spinner-container").hide();
});