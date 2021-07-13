var lastModifiedQuiz = 1;


class Quiz {
    constructor(cards, title, fullPage, instanceId, randomOrder) {
        this.cards = cards;
        this.title = title;
        this.fullPage = fullPage;
        this.instanceId = instanceId;

        this.noteIndex = 0;
        this.grades = [];
        this.hasGraded = [];
        let realIndexes = [];
        for (let i = 0; i < this.cards.length; i++) {
            this.grades.push(0);
            this.hasGraded.push(false);
            realIndexes.push(i);
        }

        if (randomOrder) {
            this.mapFakeIndexToRealIndex = realIndexes
                .map((a) => ({ sort: Math.random(), value: a }))
                .sort((a, b) => a.sort - b.sort)
                .map((a) => a.value)
        } else {
            this.mapFakeIndexToRealIndex = realIndexes;
        }

        this.getElt("previous-button").addEventListener('click', (e) => this.changePrevious(e));
        this.getElt("after-button").addEventListener('click', (e) => this.changeAfter(e));
        this.getElt("ankiapp").addEventListener('click', (e) => this.showBackCard(e));

        this.rootApp = this.getElt("ankiapp");
    }

    getElt(eltId) {
        let elt = document.getElementById(eltId + "-" + this.instanceId);
        if (elt == null) {
            throw "Id not found: " + eltId + "-" + this.instanceId;
        }
        return elt;
    }

    hasFocus() {
        return this.fullPage || (lastModifiedQuiz == this.instanceId);
    }

    launch() {
        document.onkeydown = (e) => this.changeCard(e);
        this.displayCard();
    }

    resetCards() {
        this.noteIndex = 0;
        this.grades.fill(0, 0, this.grades.length);
        this.hasGraded.fill(false, 0, this.hasGraded.length);
        $('#help-modal').modal('hide');
        this.computeGlobalGrade();
        this.displayCard();
    }


    showBackCard() {
        if (this.getElt("card-global-back").style.visibility == "visible" && !this.hasGraded[this.noteIndex]) {
            this.getElt("card-global-back").style.visibility = "hidden";

            this.getElt("card-helper").style.display = "block";
        } else {
            this.getElt("card-global-back").style.visibility = "visible";

            this.getElt("card-helper").style.display = "none";
        }

        if (!this.fullPage) {
            this.hasGraded[this.noteIndex] = true;
        }

        if (this.noteIndex < this.cards.length - 1) {
            this.getElt("after-button").style.visibility = "visible";
        }


        lastModifiedQuiz = this.instanceId;
    }

    gradCard(is_success) {
        if (this.fullPage || this.hasGraded[noteIndex]) {
            return
        }

        if (is_success) {
            this.grades.fill(1, this.noteIndex, this.noteIndex + 1);
            document.getElementById("btn-success").classList.remove("btn-outline-success");
            document.getElementById("btn-success").classList.add("btn-success");
        } else {
            this.grades.fill(0, this.noteIndex, this.noteIndex + 1);
            document.getElementById("btn-fail").classList.remove("btn-outline-danger");
            document.getElementById("btn-fail").classList.add("btn-danger");
        }
        this.hasGraded.fill(true, this.noteIndex, this.noteIndex + 1);
        this.getElt("after-button").style.visibility = "visible";

        this.computeGlobalGrade();
    }

    computeGlobalGrade() {
        let elt = document.getElementById("card-grade-value");

        let grade = this.grades.reduce((a, b) => a + b, 0) / this.grades.length;
        grade = Math.ceil(100 * grade);

        elt.innerHTML = grade;
    }

    displayCard() {
        this.getElt("after-button").style.visibility = "hidden";
        this.getElt("card-global-back").style.visibility = "hidden";

        let card = this.getElt("card");

        let realIndex = this.mapFakeIndexToRealIndex[this.noteIndex];
        this.getElt("card-front").innerHTML = this.cards[realIndex].Front;
        this.getElt("card-back").innerHTML = this.cards[realIndex].Back;

        if (this.noteIndex > 0) {
            this.getElt("previous-button").style.visibility = "visible";
        } else {
            this.getElt("previous-button").style.visibility = "hidden";
        }

        if (this.hasGraded[this.noteIndex]) {
            this.getElt("card-global-back").style.visibility = "visible";
            if (this.noteIndex < this.cards.length - 1) {
                //this.getElt("after-button").style.visibility = "visible";
            } else {
                this.getElt("after-button").style.visibility = "hidden";
            }
            this.getElt("card-helper").style.display = "none";
        } else {
            this.getElt("card-global-back").style.visibility = "hidden";
            this.getElt("after-button").style.visibility = "hidden";
            this.getElt("card-helper").style.display = "block";
        }

        let images = card.getElementsByTagName("img");
        for (let i = 0; i < images.length; i++) {
            let key = decodeURI(images[i].src.split('/').pop());
            images[i].src = "imgs/" + key;
            images[i].style = 'height: 100% width: 80%; object-fit: contain';
        };

        try {
            MathJax.typesetPromise()
        } catch (error) {
            console.log("MathJax error (used for math formula rendering): " + error)
        }
        //console.log("disp", this.getElt("card-global-back").style.visibility)

    }

    changePrevious() {
        if (this.noteIndex > 0) {
            this.noteIndex = this.noteIndex - 1;
            this.displayCard();
        }

        lastModifiedQuiz = this.instanceId;
    }

    changeAfter(e) {
        if (this.noteIndex < this.cards.length - 1 && this.hasGraded[this.noteIndex]) {
            if (!this.fullPage) {
                this.hasGraded[this.noteIndex] = true;
            }

            this.noteIndex = this.noteIndex + 1;
            console.log("next", this.getElt("card-global-back").style.visibility)
            this.displayCard();
        }
        e.stopPropagation();
        e.preventDefault();

        lastModifiedQuiz = this.instanceId;
    }

    changeCard(e) {
        if (!this.hasFocus()) {
            return;
        }

        e = e || window.event;

        if (e.keyCode == '38' || e.keyCode == '40') {
            // up arrow or down arrow
            this.showBackCard(e)
        }
        else if (e.keyCode == '37') {
            // left arrow
            this.changePrevious(e);
        }
        else if (e.keyCode == '39') {
            // right arrow
            this.changeAfter(e);
        }
        else if (e.keyCode == '81' && this.getElt("card-global-back").style.visibility == "visible") {
            // q
            this.gradCard(false);
        }
        else if (e.keyCode == '87' && this.getElt("card-global-back").style.visibility == "visible") {
            // w
            this.gradCard(true);
        }

    }

};

function launchQuiz(quizTitle, cards, randomOrder, instanceId, fullPage) {
    let quiz = new Quiz(cards, quizTitle, fullPage, instanceId, randomOrder);
    quiz.launch();
}
