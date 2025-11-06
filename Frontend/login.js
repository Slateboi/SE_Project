const loginForm = document.getElementById('loginForm');

loginForm.addEventListener('submit', (e) => {
  e.preventDefault();
  
  const username = document.getElementById('username').value;
  const password = document.getElementById('password').value;

  if(username && password){
    // Use relative path to index.html
    window.location.href = "./index.html";
  } else {
    alert("Please enter username and password");
  }
});