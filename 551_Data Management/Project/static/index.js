
function show(total,active){
    showTab=active
    for(var i=1;i<total+1;i++){
        $("#tab"+i).removeClass("tab-title-active");
        $("#content"+i).hide();
    }
    $("#tab"+active).addClass("tab-title-active");
    $("#content"+active).show();
}



//$(document).ready(function() {
//    $('form').on('submit', function(event) {
//        searchFunction()
//        event.preventDefault();
//    });
//});

function searchFunction(){
    var query = document.getElementById('inputValue').value;
    var node_method = document.getElementById('search-mr');
    var method = node_method.value;
    var node_source = document.getElementById('search-db-search');
    var source = node_source.value;
    $.ajax({
        url:"/getQueryResult",
        type:"get",
        data: 'q='+query+'&method='+method+'&source='+source,
        dataType:'text',
        processData: false,
        contentType:false,
        success:function (data){
            //Use developer tool in browser to check the data you receive
            console.log(data)

            document.getElementById("search-result").innerHTML=(
                '<label>Your query result:</label>' +
                '<div class="resultBox" id="queryResult">'+data+'</div>'
            );
        },
    })
}





//$(document).ready(function() {
//    $('form').on('submit', function(event) {
//        terminalCommand()
//        event.preventDefault();
//    });
//});

function terminalCommand(){
    var query = document.getElementById('tcInputValue').value;
    var db_node = document.getElementById('search-db-tc');
    var db = db_node.value;
    $.ajax({
        url:"/terminalCommand",
        type:"get",
        data: 'tc='+query+'&source='+db,
        dataType:'text',
        processData: false,
        contentType:false,
        success:function (data){
            //Use developer tool in browser to check the data you receive
            console.log(data)

            document.getElementById("terminal-command-result").innerHTML=(
                '<label>Terminal:</label>' +
                '<div class="resultBox" id="queryResult">'+data+'</div>'
            );
        },
    })
}



function analyticsPredict(){
    var algo_node = document.getElementById('algo');
    var algo = algo_node.value;

    var input_vars = "";
    document.querySelectorAll(".input-var").forEach(node => {
        var attr_name = node.getAttribute("attr")
        var attr_val = node.value
        input_vars = input_vars + "&" + attr_name + "=" + attr_val
    });

    $.ajax({
        url:"/analyticsPredict",
        type:"get",
        data: 'algo='+algo+input_vars,
        dataType:'text',
        processData: false,
        contentType:false,
        success:function (data){
            //Use developer tool in browser to check the data you receive
            console.log(data)

            document.getElementById("analytics-result-box").innerHTML=(
                'Predicted number of car accidents per 10,000 population based on input values = '+data
            );
        },
    })
}



function getStructure(){
    var db_node = document.getElementById('search-db-nav');
    var db = db_node.value;

    $.ajax({
        url:"/getStructure",
        type:"get",
        data: 'db='+db,
        dataType:'json',
        processData: false,
        contentType:false,
        success:function (data){
            //Use developer tool in browser to check the data you receive
            console.log(data)


            $(function(){
                treeOne = new DinampTreeEditor('.firstTree').setData(data);

                treeScnd = new DinampTreeEditor('.scndTree').setData(data).set({
                  extended: false,
                  radios: true,
                  editable:false
//                  oncheck: function(state, text, path) {
//                    $(".sncdOutput").text(path);
//                  }
                });

                 treeThird = new DinampTreeEditor('.thirdTree').setData(data).set({
                 checkboxes: true,
                 onchange: function(tree) {
                   $(".thirdOutput").text(JSON.stringify(tree.getData()));
                  }
                });
              });
        }
    })
}