var express = require("express");
var path = require('path');
var PORT = process.env.PORT || 5000;

var app = express()

app.use(express.static(path.join(__dirname, 'public')))
app.set('views', path.join(__dirname, 'views'))
app.set('view engine', 'html');
app.engine('html', require('ejs').renderFile);

app.get("/", function(req, res){
    res.render('index');
});

app.listen(PORT, () => {
  console.log(`Example app listening at http://localhost:${PORT}`)
})