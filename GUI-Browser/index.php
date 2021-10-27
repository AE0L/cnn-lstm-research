<?php 
include_once 'header.php'
 ?>
<body>
	<div class="d-flex justify-content-center" id="wrapper">
	<div id="page-content-wrapper">
            <!-- Page content-->
            <div class="container-fluid">
                <h1 class="mt-4">Welcome</h1>
                <form>
                    <div class="mb-3">
                        <label for="inputUsername" class="form-label">Enter Username</label>
                        <input type="text" class="form-control" id="inputUsername" aria-describedby="emailHelp" required>
                    </div><br>
                    <div class="d-grid gap-2">
                        <button type="submit" class="btn btn-primary" type="button">Save</button>
                    </div>
                </form>
            </div>
        </div>
    </div>
</body>
<?php
include_once 'footer.php'
?>
</html>
