<?php
include_once 'header.php'
?>


<html>

<head>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-1BmE4kWBq78iYhFldvKuhfTAU6auU8tT94WrHftjDbrCEXSU1oBoqyl2QvZ6jIW3" crossorigin="anonymous">
    <link rel="stylesheet" type="text/css" href="style.css">
</head>

<body>
    <div class="d-flex justify-content-center" id="wrapper">

        <!-- Page content wrapper-->
        <div id="page-content-wrapper">
            <!-- Page content-->
            <div class="container-fluid">
                <h1 class="mt-4">Date Settings</h1>
                <form>
                    <div class="mb-3">
                        <label class="form-label">Enter Start Date</label>
                        <input type="date" class="form-control" id="sdate"><br>
                        <label class="form-label">Enter End Date</label>
                        <input type="date" class="form-control" id="edate">
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