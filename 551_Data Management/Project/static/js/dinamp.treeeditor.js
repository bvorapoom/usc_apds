/**
  * Created by Francis Pogulis
  * francispo@inbox.lv
  * Created for a simple, yet functional UI
  *
  * Obviosly improvements should be made
  * jQuery is required
  */
class DinampTreeEditor {
  mainNode;     //selector
  uuid;
  contextMenu;  //jQuery node
  selectedItem; //jQuery node
  options;      //{}

  //TODO add max capacity and tags, stackabel=bool
  constructor(mainNode) {
    this.uuid = this.newuuid();
    if(!$(mainNode).hasClass("jsTree"))
      $(mainNode).addClass("jsTree");//make sure styles are accurate

    $(mainNode).attr("ui-uuid", this.uuid);
    this.mainNode = mainNode + "[ui-uuid='"+this.uuid+"']";

    //TODO translte contextmenu items
    this.contextMenu = $("<div class='jsTreeContextMenu' ui-uuid='"+this.uuid+"'><p>Add inside</p><p>Add after</p><p>Delete</p></div>");//<p>Move up</p><p>Move down</p>
    this.contextMenu.insertAfter($(this.mainNode));

    //one-off listeners:

    let jsTree = this;
    $(document).on("mousedown",function(e){
      if(!$(e.target).hasClass("afterIcon") && !$(e.target).hasClass("jsTreeContextMenu") && !($(e.target).parents(".jsTreeContextMenu").length > 0)){
         jsTree.contextMenu.hide();
      }
    });

    //initial options:
    this.options = {
      checkboxes: false,
      radios: false,
      editable: true
    };
    //this.rebindListeners();
  }

  setData(data) {
    $(this.mainNode).empty();
    data.forEach(element => this.addElement(element, $(this.mainNode)));
    //TODO Optimize this line here
    //$(this.mainNode).css({width: $(this.mainNode).width() + "px"}); //makes the width fixed, so when collaped it stays the same width
    this.rebindListeners();
    return this;
  }
  set(opts) {
    let jsTree = this;
    jsTree.options = opts;
    if(opts.extended === false) {
      $(this.mainNode+" .preIcon").each(function(){
        if($(this).hasClass("arrowDown")) $(this).addClass("arrowRotate");
      });
      $(this.mainNode+" .childGroup").hide();
    }
    if(opts.checkboxes === true) {
      $(this.mainNode+" .preIcon").each(function(){
        $(this).removeClass("arrowDown");
        $(this).addClass("checkboxIcon");
      });
      jsTree.options.radios = false;
    } else if(opts.radios === true) {
      $(this.mainNode+" .preIcon").each(function(){
        if(!$(this).hasClass("arrowDown")) {
          $(this).addClass("radiobtnIcon");
        }
      });
      jsTree.options.checkboxes = false;
    } else {
      jsTree.options.radios = false;
      jsTree.options.checkboxes = false;
    }

    if(opts.editable === false) {
      $(this.mainNode+" p").removeAttr("contenteditable");
      $(this.mainNode+" .afterIcon").hide();
    } else {
      jsTree.options.editable = true;
    }

    this.rebindListeners();
    return this;
  }

  getData(){
    let jsTree = this;
    var retVal = [];
    $(this.mainNode).children().each(function(){
      jsTree.pushData(retVal, jsTree, $(this));
    });
    return retVal;
  }

  pushData(parentData, jsTree, subject) {
    if(subject.is("ul")) return;
    if(subject.is(".itemParent")) {
      let currentItem = {
        title: subject.find("p").text()
      };
      if(subject.find(".preIcon").hasClass("checked")) currentItem.checked = true;
      if(subject.next().is("ul")) {
        currentItem.children = [];
        $(subject.next()).children().each(function(){
          jsTree.pushData(currentItem.children, jsTree, $(this).find(".itemParent").eq(0));
        });
      }
      parentData.push(currentItem);
    }
  }

  addElement(el, parentNode = null) {
    var $newNode;
    if(parentNode.is("ul"))
      $newNode = $("<li class='item'><div class='itemParent'><span class='preIcon'></span><p contenteditable='true'>"+el.title+"</p><span class='afterIcon'></span></div></li>");
    else
      $newNode = $("<div class='itemParent'><span class='preIcon'></span><p contenteditable='true'>"+el.title+"</p><span class='afterIcon'></span></div>");
    //if(parentNode == null) parentNode = $(this.mainNode);
    parentNode.append($newNode);
    if(el.checked === true || el.checked === "true") $newNode.find(".preIcon").addClass("checked");
    if(el.children !== undefined) {
      $newNode.find(".preIcon").addClass("arrowDown");
      var $chContainer = $("<ul class='childGroup'></ul>");
      if(parentNode.is("ul"))
        $newNode.append($chContainer);
      else
        $(this.mainNode).append($chContainer);
      el.children.forEach(element => this.addElement(element, $chContainer));
    }
  }


//EVENT LISTENERS BEGIN HERE
  unbindListenders() {
    $(this.mainNode+" p").off();
    $(this.mainNode+" .preIcon").off();
    $(this.mainNode+" .afterIcon").off();
    $(".jsTreeContextMenu[ui-uuid='"+this.uuid+"'] p").off();
  }

  rebindListeners (jsTree = this) {
    jsTree.unbindListenders();
    $(this.mainNode+" p").keydown(function(e){
      if(e.keyCode == 13) {//code here is duplicate from below
        jsTree.selectedItem = $(":focus").closest(".itemParent");
        if(jsTree.selectedItem.parent().is("li")) {
          var $newNode = $("<li class='item'><div class='itemParent'><span class='preIcon'></span><p contenteditable='true'>New item</p><span class='afterIcon'></span></div></li>");
          $newNode.insertAfter(jsTree.selectedItem.parent());
        } else if(jsTree.selectedItem.next().length > 0 && jsTree.selectedItem.next().is(".childGroup")) {
          $newNode = $("<div class='itemParent'><span class='preIcon'></span><p contenteditable='true'>New item</p><span class='afterIcon'></span></div>");
          $newNode.insertAfter(jsTree.selectedItem.next());
        } else {
          $newNode = $("<div class='itemParent'><span class='preIcon'></span><p contenteditable='true'>New item</p><span class='afterIcon'></span></div>");
          $newNode.insertAfter(jsTree.selectedItem);
        }
        jsTree.rerender(jsTree);
        return false;
      }
    });

    $(this.mainNode+" p").on('blur', function () {
      jsTree.options.onchange(jsTree);
    });

    $(this.mainNode+" .preIcon").on('click', function(){
      if($(this).hasClass("arrowDown") && !$(this).hasClass("arrowRotate")) { //children are expanded must retract
        if($(this).parent().parent().is("li"))
          $(this).parent().parent().find(".childGroup").eq(0).animate({height: "toggle"}, 400);
        else
          $(this).parent().next().animate({height: "toggle"}, 400);
        $(this).addClass("arrowRotate");
      } else if($(this).hasClass("arrowDown") && $(this).hasClass("arrowRotate")) { //children are retracted
        if($(this).parent().parent().is("li"))
          $(this).parent().parent().find(".childGroup").eq(0).animate({height: "toggle"}, 400);
        else
          $(this).parent().next().animate({height: "toggle"}, 400);
        $(this).removeClass("arrowRotate");
      } else if($(this).hasClass("checkboxIcon")) {
        if($(this).hasClass("checked")) $(this).removeClass("checked");
        else $(this).addClass("checked");
      } else if($(this).hasClass("radiobtnIcon")) {
        $(jsTree.mainNode+" .preIcon").removeClass("checked");
        $(this).addClass("checked");
      }

      if($(this).hasClass("checkboxIcon") || $(this).hasClass("radiobtnIcon")) {
        if(jsTree.options.oncheck !== undefined) {
          let pathToDis = [];
          var curItem = $(this).parent();
          while(curItem.parent().is("li") || curItem.parent().is(jsTree.mainNode)) {
            pathToDis.unshift(curItem.find("p").text());
            curItem = curItem.parent().parent().prevAll().eq(0);
          }
          jsTree.options.oncheck($(this).hasClass("checked"), $(this).parent().find("p").text(), pathToDis);
        }
        if(jsTree.options.onchange !== undefined) {
          jsTree.options.onchange(jsTree);
        }
      }
    });

    $(this.mainNode+" .afterIcon").on('click', function(){
      jsTree.contextMenu.css({
        display: "block"
      });
      var newTop;
      var newLeft;

      if($(this).offset().left + jsTree.contextMenu.width() <= $(window).width()) {
        newLeft = $(this).offset().left;
      } else {
        newLeft = $(this).offset().left - jsTree.contextMenu.width();
      }
      if($(this).offset().top + jsTree.contextMenu.height() <= $(window).height()) {
        newTop = $(this).offset().top;
      } else {
        newTop = $(this).offset().top - jsTree.contextMenu.height();
      }

      jsTree.contextMenu.css({
          top: newTop + "px",
          left: newLeft + "px" //pageX  - $(".scrollwrappermobile").scrollLeft()
      });

      jsTree.selectedItem = $(this).parent();
    });
    $(".jsTreeContextMenu[ui-uuid='"+this.uuid+"'] p").on('click', function(){
      if($(this).index() == 0) {//add inside
        var $newNode = $("<li class='item'><div class='itemParent'><span class='preIcon'></span><p contenteditable='true'>New item</p><span class='afterIcon'></span></div></li>");
        if(jsTree.options.maxLevels !== undefined) {
          if(jsTree.selectedItem.parents().filter(".childGroup").length >= jsTree.options.maxLevels - 1){
            console.log("Max levels achived");
            return;
          }
        }
        if(jsTree.selectedItem.next().length > 0 && jsTree.selectedItem.next().is(".childGroup")) {
          jsTree.selectedItem.next().append($newNode);
          if(!jsTree.selectedItem.next().is(':visible')) {
            jsTree.selectedItem.next().animate({height: "toggle"}, 400);
            jsTree.selectedItem.find(".preIcon").removeClass("arrowRotate");
          }
        } else {
          var $chContainer = $("<ul class='childGroup'></ul>");
          $chContainer.insertAfter(jsTree.selectedItem);
          $chContainer.append($newNode);
        }
      } else if($(this).index() == 1) { //add after
        if(jsTree.selectedItem.parent().is("li")) {
          var $newNode = $("<li class='item'><div class='itemParent'><span class='preIcon'></span><p contenteditable='true'>New item</p><span class='afterIcon'></span></div></li>");
          $newNode.insertAfter(jsTree.selectedItem.parent());
        } else if(jsTree.selectedItem.next().length > 0 && jsTree.selectedItem.next().is(".childGroup")) {
          $newNode = $("<div class='itemParent'><span class='preIcon'></span><p contenteditable='true'>New item</p><span class='afterIcon'></span></div>");
          $newNode.insertAfter(jsTree.selectedItem.next());
        } else {
          $newNode = $("<div class='itemParent'><span class='preIcon'></span><p contenteditable='true'>New item</p><span class='afterIcon'></span></div>");
          $newNode.insertAfter(jsTree.selectedItem);
        }
      } else if($(this).index() == 2) { //delete
        jsTree.selectedItem.css({background:"red"});
        setTimeout(() => { //waits for the animation of red to finish
          if(jsTree.selectedItem.parent().is("li")) {
            if(jsTree.selectedItem.parent().parent().children().length > 1)
              jsTree.selectedItem.parent().remove(); //deletes li
            else
              jsTree.selectedItem.parent().parent().remove(); //deletes ul
          } else if(jsTree.selectedItem.next().length > 0 && jsTree.selectedItem.next().is(".childGroup")) {
            jsTree.selectedItem.next().remove();
            jsTree.selectedItem.remove();
          } else
            jsTree.selectedItem.remove();

          if($(jsTree.mainNode).children().length == 0) { //dont leave it empty
            var $newNode = $("<div class='itemParent'><span class='preIcon'></span><p contenteditable='true'>New item</p><span class='afterIcon'></span></div>");
            $(jsTree.mainNode).append($newNode);
          }
        }, 300);
      } else if($(this).index() == 3) { //moveup

      } else if($(this).index() == 4) { //movedown

      }

      jsTree.contextMenu.hide();
      setTimeout(() => {jsTree.rerender(jsTree);}, 400); //rebind listeners when all animations and everything is done
    });
  }

  rerender(jsTree = this) {
    if(jsTree.options.checkboxes === true) {
      $(jsTree.mainNode+" .preIcon").each(function(){
        if(!$(this).hasClass("arrowDown")) {
          $(this).addClass("checkboxIcon");
        }
      });
      jsTree.options.radios = false;
    } else if(jsTree.options.radios === true) {
      $(jsTree.mainNode+" .preIcon").each(function(){
        if(!$(this).hasClass("arrowDown")) {
          $(this).addClass("radiobtnIcon");
        }
      });
    } else {
      $(jsTree.mainNode+" .itemParent").each(function(){//TODO optimize, when delete delay required, otherwise not
          if($(this).next().is("ul")){
            if($(this).next().children().length > 0) {
              $(this).find(".preIcon").eq(0).addClass("arrowDown");
              if($(this).next().is(":visible"))
                $(this).find(".preIcon").eq(0).removeClass("arrowRotate");
              else
                $(this).find(".preIcon").eq(0).addClass("arrowRotate");
            } else
              $(this).find(".preIcon").eq(0).removeClass("arrowDown");
          } else
            $(this).find(".preIcon").eq(0).removeClass("arrowDown");
      });
    }
    if(jsTree.options.onchange !== undefined) {
      jsTree.options.onchange(jsTree);
    }

    jsTree.rebindListeners(jsTree);
  }

  newuuid() {
    return ([1e7]+-1e11).replace(/[018]/g, c =>
      (c ^ crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> c / 4).toString(16)
    );
  }
}
