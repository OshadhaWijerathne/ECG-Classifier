// Here the required credentials are imported from the .env file to keep them secure. .env file is not commited to git usually.
// Include it in gitignore.

const express = require("express");
const app = express();
const bodyParser = require("body-parser");
const cors = require("cors"); // Import the CORS middleware
const mysql = require("mysql");

app.use(bodyParser.json()); // Parse JSON requests
app.use(cors()); // Enable CORS for all routes

require("dotenv").config();
const DB_HOST = process.env.DB_HOST;
const DB_USER = process.env.DB_USER;
const DB_PASSWORD = process.env.DB_PASSWORD;
const DB_DATABASE = process.env.DB_DATABASE;
const DB_PORT = process.env.DB_PORT;
const db = mysql.createPool({
  connectionLimit: 100,
  host: DB_HOST,
  user: DB_USER,
  password: DB_PASSWORD,
  database: DB_DATABASE,
  port: DB_PORT,
});

// This function obtains a connection from the pool.
db.getConnection((err, connection) => {
  if (err) throw err;
  console.log("DB connected successful:" + connection.threadId);
});

// Get the express JS server up and running.
const port = process.env.PORT;
app.listen(port, () => console.log(`Server Started on port ${port}...`));

// Adding a route to Create a User ( createUser)
const bcrypt = require("bcrypt");
app.use(express.json());

//middleware to read req.body.<params>
//CREATE USER
app.post("/createUser", async (req, res) => {
  const user = req.body.user;
  const hashedPassword = await bcrypt.hash(req.body.pwd, 10);
  db.getConnection(async (err, connection) => {
    if (err) throw err;
    const sqlSearch = "SELECT * FROM userTable WHERE user = ?";
    const search_query = mysql.format(sqlSearch, [user]);
    const sqlInsert = "INSERT INTO userTable VALUES (0,?,?)";
    const insert_query = mysql.format(sqlInsert, [user, hashedPassword]);
    // ? will be replaced by values
    // ?? will be replaced by string
    await connection.query(search_query, async (err, result) => {
      if (err) throw err;
      console.log("------> Search Results");
      console.log(result.length);
      if (result.length != 0) {
        connection.release();
        console.log("------> User already exists");
        res.json({ status: false, message: "User already exists" });
      } else {
        await connection.query(insert_query, (err, result) => {
          connection.release();
          if (err) throw err;
          console.log("--------> Created new User");
          console.log(result.insertId);
          res.json({ status: true, message: "Created a new User" });
        });
      }
    }); //end of connection.query()
  }); //end of db.getConnection()
}); //end of app.post()

//Authentication of a User ( Login Authentication )
app.post("/login", (req, res) => {
  const user = req.body.user;
  const password = req.body.pwd;
  db.getConnection(async (err, connection) => {
    if (err) throw err;
    const sqlSearch = "Select * from userTable where user = ?";
    const search_query = mysql.format(sqlSearch, [user]);
    await connection.query(search_query, async (err, result) => {
      connection.release();

      if (err) throw err;
      if (result.length == 0) {
        console.log("--------> User does not exist");
        res.json({ status: false, message: "user does not exist" });
      } else {
        const hashedPassword = result[0].password;
        //get the hashedPassword from result
        if (await bcrypt.compare(password, hashedPassword)) {
          console.log("---------> Login Successful");
          // res.sendStatus(200); //.send({ message: `${user} is logged in!` });
          res.json({ status: true, data: { user: "login" } });
        } else {
          console.log("---------> Password Incorrect");
          // res.sendStatus(404); //.send({ message: "Password incorrect!" });
          res.json({ status: false, message: "password incorect" });
        }
      }
    });
  });
});
