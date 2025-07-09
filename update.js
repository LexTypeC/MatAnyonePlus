module.exports = {
  run: [{
    method: "shell.run",
    params: {
      message: "git pull"
    }
  }, {
    method: "shell.run",
    params: {
      path: "app",
      message: "git pull"
    }
  }, {
    method: "fs.copy",
    params: {
      src: "app.py",
      dest: "app/hugging_face/app.py"
    }
  }, {
    method: "fs.copy",
    params: {
      src: "matanyone_wrapper.py",
      dest: "app/hugging_face/matanyone_wrapper.py"
    }
  },
 ]
}
