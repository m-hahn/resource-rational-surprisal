function make_slides(f) {
    var slides = {};

    slides.consent = slide({
        name: "consent",
        start: function() {
            exp.startT = Date.now();
            $("#consent_2").hide();
            exp.consent_position = 0;
        },
        button: function() {
            exp.go(); //use exp.go() if and only if there is no "present" data.
        }
    });



    slides.i0 = slide({
        name: "i0",
        start: function() {
            exp.startT = Date.now();
        }
    });

    slides.instructions1 = slide({
        name: "instructions1",
        start: function() {
            $(".instruction_condition").html("Between subject intruction manipulation: " + exp.instruction);
        },
        button: function() {
            exp.go(); //use exp.go() if and only if there is no "present" data.
        }
    });


/////////////////////////////////////
    slides.rest = slide({
        name: "rest",
        start: function() {
		msg = ""
		exp.rest_phase += 1;
		if(exp.errors > 6.25) {
			msg = "You have been making more errors than most other participants. Please try to be more accurate in your responses."
		} else if(exp.errors > 5.0) {
			msg = "Most participants make fewer errors. Try to be more accurate in your responses."
		} else if(exp.errors > 3.25) {
			msg = "You are doing okay, though the majority of participants make fewer errors."
		} else if(exp.errors < 1.25) {
			msg = "You are doing very well!"
		} else {
			msg = "You are doing well. This number of errors is similar to most other participants."
		}
            $(".error-info").html("You have made " + exp.errors + " errors. "+msg);
		exp.errors=0;
        },
        button: function() {
            exp.go(); //use exp.go() if and only if there is no "present" data.
        }
    });




    ////////////////////////////////////////////


    for (mazeGroup = 1; mazeGroup <= 4; mazeGroup++) {
        slides["maze" + mazeGroup] = slide({
            name: "maze",
            present: stimuliContextPart[mazeGroup],
            present_handle: function(stim) {

                console.log(stim);
                this.stim = stim;
                console.log(stim);
                this.words = this.stim.s.split(" ")
                this.alts = this.stim.a.split(" ")
                this.order = [];
                this.mazeResults = [];
                this.correct = [];
                for (i = 0; i < this.words.length; i++) {
                    this.order.push(_.sample([0, 1], 1)[0]);
                    this.mazeResults.push([null, null]);
                    this.correct.push(null);
                }
                words = ["hallo"]
                this.redo = true; // redo when people make an error
                var t = this;
                var repeat = true;
                this.currentWord = 0;
                this.stoppingPoint = this.words.length
                console.log(stim.r);
                if (stim.r != null) {
                    this.regions = stim.r.split(" ");
                } else {
                    this.regions = this.words;
                }
                console.log(this.regions);
                this.listener = function(event) {
                    console.log(event);
                    console.log(t.currentWord);
                    var time = new Date().getTime();
                    var code = event.keyCode;
                    if (t.currentWord == -1) {
                        console.log("Current word -1 ");
                        t.currentWord = 0;
                        t.showWord(t.currentWord);
                        $(".Maze-error").html("");
                        if (t.redo) {
                            console.log("REPEATING");
                            $(".Maze-lword").show();
                            $(".Maze-rword").show();
                            $(".Maze-larrow").show();
                            $(".Maze-rarrow").show();
                            return;
                        } else {
                            t.button();
                            return;
                        }
                    } else if (t.currentWord == -2) {
                        t.button();
                        return;
                    } else if ((code == 69 || code == 73) && (!((code == 69 && t.order[t.currentWord] == 0) || (code == 73 && t.order[t.currentWord] == 1))) && t.currentWord == 0) {
                        console.log("Do nothing");
                    } else if (code == 69 || code == 73) {
                        var word = t.currentWord;
                        if (word <= t.stoppingPoint) {
                            correct = ((code == 69 && t.order[word] == 0) || (code == 73 && t.order[word] == 1)) ? "yes" : "no";
                            if (t.correct[word] == null) {
                                t.mazeResults[word][0] = time;
                                t.mazeResults[word][1] = t.previousTime;
                                t.correct[word] = correct
                                console.log(t.mazeResults);
                                console.log(t.correct);
                            }
                            if (correct == "no" & t.redo) {
				    if(t.stim.practice != true) {
   				         exp.errors += 1;
   				         exp.total_errors += 1;
				    }
                                $(".Maze-error").html("Incorrect. Please try again.");
 //                               $(".Maze-lword").hide();
   //                             $(".Maze-rword").hide();
     //                           $(".Maze-larrow").hide();
       //                         $(".Maze-rarrow").hide();
//                                t.currentWord = -1;
                                return true;
                            } else if (correct == "no") {
                                $(".Maze-error").html("Incorrect! Press any key to continue");
                                $(".Maze-lword").hide();
                                $(".Maze-rword").hide();
                                $(".Maze-larrow").hide();
                                $(".Maze-rarrow").hide();
                                t.currentWord = -1;
                                return true;
                            } else if (correct == "yes") {
                                $(".Maze-error").html("");
                            }
                        }
                        t.previousTime = time;
                        ++(t.currentWord);
                        if (t.currentWord >= t.stoppingPoint) {
                            $(".Maze-counter").html("Correct! Press any key to continue.");
                            $(".Maze-lword").hide();
                            $(".Maze-rword").hide();
                            $(".Maze-larrow").hide();
                            $(".Maze-rarrow").hide();
                            t.currentWord = -2;
                            return true;
                        }
                        t.showWord(t.currentWord);
                        return false;
                    } else {
                        return true;
                    }
                }
                document.addEventListener('keydown', this.listener);
                this.showWord(0);

                $(".Maze-lword").show();
                $(".Maze-rword").show();
                $(".Maze-larrow").show();
                $(".Maze-rarrow").show();
            },

            showWord: function(w) {
                if (this.currentWord < this.stoppingPoint) {
                    $(".Maze-lword").html((this.order[this.currentWord] === 0) ?
                        this.words[this.currentWord] : this.alts[this.currentWord]);
                    $(".Maze-rword").html((this.order[this.currentWord] === 0) ?
                        this.alts[this.currentWord] : this.words[this.currentWord]);
                    exp.wordsSoFar++;
                    $(".Maze-counter").html("Words so far: " + exp.wordsSoFar);
                    this.previousTime = new Date().getTime();
                }
            },

            button: function() {
                console.log("CALL BUTTON");
                document.removeEventListener('keydown', this.listener);
                this.log_responses();
                _stream.apply(this); //use exp.go() if and only if there is no "present" data.
            },

            init_sliders: function() {
                utils.make_slider("#slider0_ctxt", function(event, ui) {
                    exp.sliderPost = ui.value;
                });
            },
            log_responses: function() {
                document.removeEventListener('keydown', this.listener);
                console.log(this.words);
                console.log(this.mazeResults);
                byWords = [];
                for (i = 0; i < this.words.length; i++) {
                    byWords.push({
                        "word": this.words[i],
                        "rt": this.mazeResults[i][0] - this.mazeResults[i][1],
                        "correct": this.correct[i],
                        "region": this.regions[i],
                        "alt": this.alts[i]
                    })
                    console.log(byWords[byWords.length - 1]);
                }
                console.log(byWords);
                dataForThisTrial = {
                    "sentence": this.stim.s,
                    "item": this.stim.item,
                    "condition": this.stim.condition,
                    "byWords": byWords,
                    "noun": this.stim.noun,
                    "distractor_condition": this.stim.distractor_condition,
                    "slide_number": exp.phase
                };
                exp.data_trials.push(dataForThisTrial);
                console.log(exp.data_trials[exp.data_trials.length - 1]);

		    // UPLOADING RESULTS
            },
        })
    };



// Subject Information
    slides.subj_info = slide({
        name: "subj_info",
        submit: function(e) {
            //if (e.preventDefault) e.preventDefault(); // I don't know what this means.
            exp.subj_data = {
                language: $("#language").val(),
                enjoyment: $("#enjoyment").val(),
                asses: $('input[name="assess"]:checked').val(),
                comments: $("#comments").val(),
                suggested_pay: $("#suggested_pay").val(),
		    errors : exp.total_errors//,
//		    conditions : exp.conditions_chosen
            };
		console.log(exp.subj_data);
            exp.go(); //use exp.go() if and only if there is no "present" data.
        }
    });

    slides.thanks = slide({
        name: "thanks",
        start: function() {

          // UPLOADING RESULTS
        }
    });

    return slides;
}

/// init ///
function init() {
    repeatWorker = false;

    exp.current_score_click = 0;
    exp.total_quiz_trials_click = 0;

    exp.current_score = 0;
    exp.total_quiz_trials = 0;
    exp.hasDoneTutorialRevision = false;
    exp.shouldDoTutorialRevision = false;
    exp.hasEnteredInterativeQuiz = false;

    exp.trials = [];
    exp.catch_trials = [];
    exp.instruction = _.sample(["instruction1", "instruction2"]);
    exp.system = {
        Browser: BrowserDetect.browser,
        OS: BrowserDetect.OS,
        screenH: screen.height,
        screenUH: exp.height,
        screenW: screen.width,
        screenUW: exp.width
    };

    //blocks of the experiment:
    exp.structure = [];
    exp.structure.push('i0')
    exp.structure.push('consent')
    exp.structure.push('instructions1')
    exp.structure.push('maze1')
    exp.structure.push('rest')
    exp.structure.push('maze2')
    exp.structure.push('rest')
    exp.structure.push('maze3')
    exp.structure.push('rest')
    exp.structure.push('maze4')

    exp.structure.push('subj_info')
    exp.structure.push('thanks');


    exp.data_trials = [];
    //make corresponding slides:
    exp.slides = make_slides(exp);

    exp.nQs = utils.get_exp_length(); //this does not work if there are stacks of stims (but does work for an experiment with this structure)
    //relies on structure and slides being defined

    $('.slide').hide(); //hide everything

    //make sure turkers have accepted HIT (or you're not in mturk)
    $("#start_button").click(function() {
        if (turk.previewMode) {
            $("#mustaccept").show();
        } else {
            $("#start_button").click(function() {
                $("#mustaccept").show();
            });
            exp.go();
        }
    });

    exp.order_questionnaires = _.sample([
        [0, 1],
        [1, 0]
    ])

    exp.wordsSoFar = 0;
	exp.total_errors = 0;
	exp.errors = 0;
	exp.rest_phase=0;

    exp.go(); //show first slide
}
