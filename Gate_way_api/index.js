const express =require('express');
const app =express();
app.get('/',(req,res)=>{
    res.send('vanakam da mapla backendlarunthu .......')
})
app.get('/login',(req,res)=>{
    res.send('please login bro .........')
})

app.listen(5000,()=>{console.log("I'm alive broo..........");
})
