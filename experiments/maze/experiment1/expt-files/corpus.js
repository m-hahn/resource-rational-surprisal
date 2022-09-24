// https://tc39.github.io/ecma262/#sec-array.prototype.includes
if (!Array.prototype.includes) {
  Object.defineProperty(Array.prototype, 'includes', {
    value: function(searchElement, fromIndex) {

      // 1. Let O be ? ToObject(this value).
      if (this == null) {
        throw new TypeError('"this" is null or not defined');
      }

      var o = Object(this);

      // 2. Let len be ? ToLength(? Get(O, "length")).
      var len = o.length >>> 0;

      // 3. If len is 0, return false.
      if (len === 0) {
        return false;
      }

      // 4. Let n be ? ToInteger(fromIndex).
      //    (If fromIndex is undefined, this step produces the value 0.)
      var n = fromIndex | 0;

      // 5. If n ≥ 0, then
      //  a. Let k be n.
      // 6. Else n < 0,
      //  a. Let k be len + n.
      //  b. If k < 0, let k be 0.
      var k = Math.max(n >= 0 ? n : len - Math.abs(n), 0);

      function sameValueZero(x, y) {
        return x === y || (typeof x === 'number' && typeof y === 'number' && isNaN(x) && isNaN(y));
      }

      // 7. Repeat, while k < len
      while (k < len) {
        // a. Let elementK be the result of ? Get(O, ! ToString(k)).
        // b. If SameValueZero(searchElement, elementK) is true, return true.
        // c. Increase k by 1. 
        if (sameValueZero(o[k], searchElement)) {
          return true;
        }
        k++;
      }

      // 8. Return false
      return false;
    }
  });
}

var stimuliContext =  makeStimsContext();
var stimuliContextPart = {};
stimuliContextPart[1] = stimuliContext.slice(0, Math.floor(stimuliContext.length/4))
stimuliContextPart[2] = stimuliContext.slice(Math.floor(stimuliContext.length/4), Math.floor(2*stimuliContext.length/4))
stimuliContextPart[3] = stimuliContext.slice(Math.floor(2*stimuliContext.length/4), Math.floor(3*stimuliContext.length/4))
stimuliContextPart[4] = stimuliContext.slice(Math.floor(3*stimuliContext.length/4))

console.log("@@@");
console.log(stimuliContextPart[1].length);
console.log(stimuliContextPart[2].length);
console.log(stimuliContextPart[3].length);
console.log(stimuliContextPart[4].length);
console.log(stimuliContext.length);

console.log(stimuliContextPart[1]);
console.log(stimuliContextPart[2]);
console.log(stimuliContextPart[3]);
console.log(stimuliContextPart[4]);




function makeStimsContext() {
     console.log("MAKE STIMS CONTEXT")



     function addStimulus(stimsHere, item, condition, sentence, question, answer) {
          stimulus = {
		      "item" : item,
		      "condition" : condition,
		      "sentence" : sentence,
		      "question" : question,
		      "answer" : answer
                 };
          stimsHere.push(stimulus);
     }
     fillers = [];
     stimsExplanation = [];
     stims = [];
     stimsTraining = [];
	critical = [];
     

     conditionAssignment = [];
     conditionsCounts = [0, 0, 0, 0];
	if(Math.random() > 0.5) {
		cond1 = 0;
		cond2 = -1;
		cond3 = 0;
		cond4 = -1;
	} else {
		cond1 = -1;
		cond2 = 0;
		cond3 = -1;
		cond4 = 0;
	}
//     for(var i=0; i<3; i++) {
     	   conditionAssignment.push(1)
     	   conditionAssignment.push(cond1)
     	   conditionAssignment.push(cond3)
     	   conditionAssignment.push(cond2)
     	   conditionAssignment.push(cond4)
     	   conditionAssignment.push(cond1)
     	   conditionAssignment.push(cond2)
     	   conditionAssignment.push(cond3)
     	   conditionAssignment.push(cond4)
     	   conditionAssignment.push(1)
  //   }
     console.log(conditionAssignment); 
    
nouns = {}

nouns["CLAIM"] = []
nouns["ACCUSATION"] = []
nouns["FACT"] = []
nouns["HOPE"] = []
nouns["CHANCE"] = []
nouns["FEAR"] = []
nouns["PREDICTION"] = []

// CLAIM: a claim whose truth or falsity is not presupposed.
nouns["CLAIM"].push("announcement")
nouns["CLAIM"].push("assertion")
nouns["CLAIM"].push("assessment")
nouns["CLAIM"].push("assumption")
nouns["CLAIM"].push("belief")
nouns["CLAIM"].push("claim")
nouns["CLAIM"].push("conclusion")
nouns["CLAIM"].push("confirmation")
nouns["CLAIM"].push("declaration")
nouns["CLAIM"].push("feeling")
nouns["CLAIM"].push("finding")
nouns["CLAIM"].push("idea")
nouns["CLAIM"].push("indication")
nouns["CLAIM"].push("inkling")
//nouns["CLAIM"].push("insinuation") // removing this for now because the counts are really high-variance
nouns["CLAIM"].push("news")
nouns["CLAIM"].push("notion")
nouns["CLAIM"].push("opinion")
nouns["CLAIM"].push("perception")
nouns["CLAIM"].push("presumption")
nouns["CLAIM"].push("remark")
nouns["CLAIM"].push("reminder")
nouns["CLAIM"].push("revelation")
nouns["CLAIM"].push("rumor")
nouns["CLAIM"].push("speculation")
nouns["CLAIM"].push("statement")
nouns["CLAIM"].push("suggestion")
nouns["CLAIM"].push("theory")
nouns["CLAIM"].push("view")
nouns["CLAIM"].push("assurance")
nouns["CLAIM"].push("message")
nouns["CLAIM"].push("contention")
nouns["CLAIM"].push("impression")
nouns["CLAIM"].push("opinion")
nouns["CLAIM"].push("sense")
nouns["CLAIM"].push("presumption")
nouns["CLAIM"].push("revelation")
nouns["CLAIM"].push("intuition")
nouns["CLAIM"].push("conjecture")
nouns["CLAIM"].push("conviction")
nouns["CLAIM"].push("thought")
nouns["CLAIM"].push("claim")
nouns["CLAIM"].push("conclusion")
nouns["CLAIM"].push("feeling")
nouns["CLAIM"].push("finding")
nouns["CLAIM"].push("idea")
nouns["CLAIM"].push("indication")
nouns["CLAIM"].push("presumption")
nouns["CLAIM"].push("revelation")
nouns["CLAIM"].push("rumor")
nouns["CLAIM"].push("speculation")
nouns["CLAIM"].push("guess")
nouns["CLAIM"].push("story")
nouns["CLAIM"].push("report")



nouns["ACCUSATION"].push("admission")
nouns["ACCUSATION"].push("allegation")
nouns["ACCUSATION"].push("accusation")
nouns["ACCUSATION"].push("insinuation") 
nouns["ACCUSATION"].push("complaint")
nouns["ACCUSATION"].push("suspicion")
//nouns["CHANCE"].push("chance") 
//nouns["CHANCE"].push("probability")
//nouns["CHANCE"].push("likelihood")
//nouns["FEAR"].push("fear")
//nouns["FEAR"].push("concern")
//nouns["HOPE"].push("reassurance")
//nouns["HOPE"].push("hope")
//nouns["HOPE"].push("promise")
//nouns["PREDICTION"].push("prediction")
//nouns["PREDICTION"].push("expectation")
nouns["FACT"].push("truth")
nouns["FACT"].push("fact")
nouns["FACT"].push("reminder")
nouns["FACT"].push("disclosure")
nouns["FACT"].push("proof")
nouns["FACT"].push("realization")
nouns["FACT"].push("observation")
nouns["FACT"].push("understanding")
nouns["FACT"].push("proof")
nouns["FACT"].push("certainty")
nouns["FACT"].push("recognition")
nouns["FACT"].push("disclosure")

nounsByThatBiasOrder = [];




nounsByThatBiasOrder = [];



// Nouns selected according to the average of the three log-frequencies
nounsByThatBiasOrder.push("story")
nounsByThatBiasOrder.push("report")
nounsByThatBiasOrder.push("assessment")
nounsByThatBiasOrder.push("truth")
nounsByThatBiasOrder.push("declaration")
nounsByThatBiasOrder.push("complaint")
nounsByThatBiasOrder.push("admission")
nounsByThatBiasOrder.push("disclosure")
nounsByThatBiasOrder.push("confirmation")
//nounsByThatBiasOrder.push("guess")
nounsByThatBiasOrder.push("remark")
nounsByThatBiasOrder.push("news")
nounsByThatBiasOrder.push("proof")
nounsByThatBiasOrder.push("message")
nounsByThatBiasOrder.push("announcement")
nounsByThatBiasOrder.push("statement")
nounsByThatBiasOrder.push("thought")
nounsByThatBiasOrder.push("allegation")
nounsByThatBiasOrder.push("indication")
nounsByThatBiasOrder.push("recognition")
nounsByThatBiasOrder.push("speculation")
nounsByThatBiasOrder.push("accusation")
nounsByThatBiasOrder.push("reminder")
nounsByThatBiasOrder.push("rumor")
nounsByThatBiasOrder.push("finding")
nounsByThatBiasOrder.push("idea")
nounsByThatBiasOrder.push("feeling")
nounsByThatBiasOrder.push("conjecture")
nounsByThatBiasOrder.push("perception")
nounsByThatBiasOrder.push("certainty")
nounsByThatBiasOrder.push("revelation")
nounsByThatBiasOrder.push("understanding")
nounsByThatBiasOrder.push("claim")
nounsByThatBiasOrder.push("view")
nounsByThatBiasOrder.push("observation")
nounsByThatBiasOrder.push("conviction")
nounsByThatBiasOrder.push("presumption")
nounsByThatBiasOrder.push("intuition")
nounsByThatBiasOrder.push("opinion")
nounsByThatBiasOrder.push("conclusion")
nounsByThatBiasOrder.push("notion")
nounsByThatBiasOrder.push("suggestion")
nounsByThatBiasOrder.push("sense")
nounsByThatBiasOrder.push("suspicion")
nounsByThatBiasOrder.push("assurance")
//nounsByThatBiasOrder.push("insinuation")
nounsByThatBiasOrder.push("realization")
nounsByThatBiasOrder.push("assertion")
nounsByThatBiasOrder.push("impression")
nounsByThatBiasOrder.push("contention")
nounsByThatBiasOrder.push("assumption")
nounsByThatBiasOrder.push("belief")
nounsByThatBiasOrder.push("fact")



// Ensure that each noun has an associated class.

topNouns = [];
for(n = 0; n<nounsByThatBiasOrder.length; n++) {
	success = false;
	for(key in nouns) {
		if(nouns[key].includes(nounsByThatBiasOrder[n])) {
			success = true;
		}
	}
	if(!success) {
	   console.log("EXCLUDED", nounsByThatBiasOrder[n], success, nouns["HOPE"]);
	}
	if(success) {
		topNouns.push(nounsByThatBiasOrder[n]);
	}
}



// Slice nouns into high- and low-embedding bias lists
console.log("------------------------")
console.log("TOP NOUNS BEFORE SLICING", topNouns);

topNouns1 = _.sample(topNouns.slice(0, 15), 10);
topNouns2 = _.sample(topNouns.slice(topNouns.length-15, topNouns.length), 10);

// Select five high-embedding bias and five low-embedding bias nouns
console.log(topNouns1);
console.log(topNouns2);
console.log("SELECTION");
topNouns = [];
for(i=0; i<5; i++) {
	selected1 = (Math.random() > 0.5) ? 2*i : 2*i+1;
	selected2 = (Math.random() > 0.5) ? 2*i : 2*i+1;
	console.log(selected1, selected2, topNouns1.length, topNouns2.length);
	topNouns.push(topNouns1[selected1]);
	topNouns.push(topNouns2[selected2]);
}


console.log("LENGTH", topNouns.length, topNouns);


// Now, here are all the stimuli.
// selected : true means the stimulus was included in the experiment, selected : false means it was not.
// After completion of stimulus designs, 32 stimuli were selected balancing different syntactic structures in the second-to-last verb phrase (e.g., was+adjective, verb+object).
// Compatible and incompatible versions were originally designed, but the experiment was only run with the incompatible versions

continuations250 = []

for(i=0; i<44; i++) {
	continuations250.push([]);
}

continuations250[0].push({ selected : false, item : "Mixed_0", s : "that the lifeguard who the children admired /was brave /excited everyone.", a : "x-x-x seeing coal nor sculpting mid sea indicate boiling map oxide crocodile function.", n : "FACT CLAIM"})
continuations250[0].push({ selected : false, item : "Mixed_0", s : "that the lifeguard who the children admired /was curious /excited everyone.", a : "x-x-x quoted anti thy rickshaws us big consists valleys map circles crocodile projects."})
continuations250[1].push({ selected : true, item : "Mixed_1", s : "that the commander who the president appointed /was fired /annoyed people.", a : "x-x-x merely easy fat somewhere net few something different gas query stamped factor.", n : "FACT CLAIM"})
continuations250[1].push({ selected : true, item : "Mixed_1", s : "that the commander who the president appointed /was ignored /annoyed people.", a : "x-x-x anyone anti eat requiring sin eye centuries dangerous try justify pleaded really."})
continuations250[2].push({ selected : true, item : "Mixed_2", s : "that the criminal who the officer arrested /was innocent /turned out to be entirely bogus.", a : "x-x-x trying real joy answered hot him evident exciting eat quarters circle ago sat got reaction blitz.", n : "CLAIM"})
continuations250[2].push({ selected : true, item : "Mixed_2", s : "that the criminal who the officer arrested /was reliable /turned out to be entirely bogus.", a : "x-x-x accept wise sky whenever why sun however juvenile eat creature circle am thy is sciences swirl."})
continuations250[3].push({ selected : false, item : "Mixed_3", s : "that the victim who the criminal assaulted /remained crippled /was unnerving.", a : "x-x-x argued duty lot anyway sin son consider inventing projects shedding sit resurrect.", n : "FACT CLAIM"})
continuations250[3].push({ selected : false, item : "Mixed_3", s : "that the victim who the criminal assaulted /remained hidden /was unnerving.", a : "x-x-x enough boat eat retain add why although loosening examples breath sit reconnect."})
continuations250[4].push({ selected : true, item : "Mixed_4", s : "that the surgeon who the nurse assisted /read the news /was not a secret.", a : "x-x-x unless real sin sounded oil car smile metallic come bit else sin thy map fields.", n : "FACT CLAIM"})
continuations250[4].push({ selected : true, item : "Mixed_4", s : "that the surgeon who the nurse assisted /made the news /was not a secret.", a : "x-x-x though coal sin sincere big ten shore husbands star buy tend thy thy ago groups."})
continuations250[5].push({ selected : true, item : "Mixed_5", s : "that the worker who the thug attacked /was rescued /surprised people.", a : "x-x-x killed ways thy liable lie sea raga explicit mid implant libraries appeal.", n : "FACT CLAIM"})
continuations250[5].push({ selected : true, item : "Mixed_5", s : "that the worker who the thug attacked /was questioned /surprised people.", a : "x-x-x myself food die gently joy map slur opinions add confession temporary review."})
continuations250[6].push({ selected : true, item : "Mixed_6", s : "that the pianist who the sponsors backed /played sonatas /pleased everyone.", a : "x-x-x myself thou nor steamed sky men branched hatred sector chopper muscles measured.", n : "FACT CLAIM"})
continuations250[6].push({ selected : true, item : "Mixed_6", s : "that the pianist who the sponsors backed /sounded hopeful /pleased everyone.", a : "x-x-x trying vast eat cheated aid eye emergent luxury despair isolate reserve products."})
continuations250[7].push({ selected : true, item : "Mixed_7", s : "that the actor who the producer blamed /got very sick /was sad to hear.", a : "x-x-x myself soul die shake ten man insights bundle arm pair mile see den arm uses.", n : "FACT CLAIM"})
continuations250[7].push({ selected : true, item : "Mixed_7", s : "that the actor who the producer blamed /made everyone cry /was sad to hear.", a : "x-x-x occurs thou nor opens net man literacy dealer mark doctrine egg sea cup mid ages."})
continuations250[8].push({ selected : true, item : "Mixed_8", s : "that the politician who the banker bribed /laundered money /came as a shock to his supporters.", a : "x-x-x accept male sin identifies red ad pulses pianos motorcade angle cent joy nor width go nor indirectly.", n : "FACT CLAIM ACCUSATION"})
continuations250[8].push({ selected : true, item : "Mixed_8", s : "that the politician who the banker bribed /disappointed people /came as a shock to his supporters.", a : "x-x-x sought rise joy faithfully add art malice cleric evolutionary inches flat thy nor ranks let fat correlated."})
continuations250[9].push({ selected : true, item : "Mixed_9", s : "that the paramedic who the swimmer called /saved the children /was good news.", a : "x-x-x remove soul sky showcased ten off sagging social tower sum yourself non sale seem.", n : "FACT CLAIM"})
continuations250[9].push({ selected : true, item : "Mixed_9", s : "that the paramedic who the swimmer called /pleased the children /was good news.", a : "x-x-x obtain evil fat immigrate dry how slicing notion savings am commerce sun sale tree."})
continuations250[10].push({selected : false, item : "Mixed_10", s : "that the extremist who the agent caught /received awards /was unfortunate.", a : "x-x-x looked duty ill placental sit sum apart weight superior slides sin proportions.", n : "FACT CLAIM"})
continuations250[10].push({selected : false, item : "Mixed_10", s : "that the extremist who the agent caught /received attention /was unfortunate.", a : "x-x-x reduce evil thy unaltered his son where degree relation objective job discoveries."})
continuations250[11].push({selected : true, item : "Mixed_11", s : "that the trader who the businessman consulted /tricked Kim /did not bother Mary.", a : "x-x-x issued jobs thy closes hot joy proficiency agreeable magnate Vie lie joy echoed Main.", n : "FACT CLAIM ACCUSATION"})
continuations250[11].push({selected : true, item : "Mixed_11", s : "that the trader who the businessman consulted /hurt Kim /did not bother Mary.", a : "x-x-x looked hear eat hereby add ice discernible rejection pick Awe thy red relics Girl."})
continuations250[12].push({selected : true, item : "Mixed_12", s : "that the mayor who the farmer disliked /was elected /did not bother the farmer.", a : "x-x-x itself cash fat enjoy men lot plasma auditing bed explain net am folder lot breath.", n : "CLAIM"})
continuations250[12].push({selected : true, item : "Mixed_12", s : "that the mayor who the farmer disliked /was refuted /did not bother the farmer.", a : "x-x-x myself coal nor thank low ice arrive cruisers ask revolts aid fat garage did fought."})
continuations250[13].push({selected : true, item : "Mixed_13", s : "that the neighbor who the woman distrusted /poisoned the dog /seemed to be a malicious smear.", a : "x-x-x remove wine sin muscular bad net aware billboards fashions mid ago oxygen boy net tax capturing tiled.", n : "CLAIM ACCUSATION"})
continuations250[13].push({selected : true, item : "Mixed_13", s : "that the neighbor who the woman distrusted /startled the child /seemed to be a malicious smear.", a : "x-x-x enough soil sky disposed sky joy doubt appointees pilgrims ice knows broken ago net ago anthology azure."})
continuations250[14].push({selected : true, item : "Mixed_14", s : "that the senator who the diplomat encountered /opposed Josh /was absolutely true.", a : "x-x-x expect fine ill connect hot aid watchful sensitivity balance Clip sky securities hear.", n : "CLAIM"})
continuations250[14].push({selected : true, item : "Mixed_14", s : "that the senator who the diplomat encountered /bothered Josh /was absolutely true.", a : "x-x-x become vast die exceeds add old gorgeous enterprises driveway Gait eye mechanisms size."})
continuations250[15].push({selected : true, item : "Mixed_15", s : "that the preacher who the parishioners fired /was lying /startled Anne.", a : "x-x-x inches pure die develops lie end transferable cubic end width builders Gods.", n : "FACT CLAIM ACCUSATION"})
continuations250[15].push({selected : true, item : "Mixed_15", s : "that the preacher who the parishioners fired /was annoying /startled Anne.", a : "x-x-x filled fine joy reminded sum thy reproducible flash sky calamity promotes Hole."})
continuations250[16].push({selected : false, item : "Mixed_16", s : "that the student who the professor hated /was lazy /gained traction in the university.", a : "x-x-x anyone salt ill achieve sky old ourselves renal gas odor worthy foreseen bed add understood.", n : "CLAIM ACCUSATION"})
continuations250[16].push({selected : false, item : "Mixed_16", s : "that the student who the professor hated /was foolish /gained traction in the university.", a : "x-x-x occurs foot die discuss why nor providing parks arm enforce lifted blending lie lie maintained."})
continuations250[17].push({selected : true, item : "Mixed_17", s : "that the counselor who the writer hired /hated the janitor /was dismissed as untrue.", a : "x-x-x obtain ways eat intervene hot war fairly blade crude map toppled oil doctrines dry vortex.", n : "CLAIM FACT"})
continuations250[17].push({selected : true, item : "Mixed_17", s : "that the counselor who the writer hired /surprised the janitor /was dismissed as untrue.", a : "x-x-x taught neck lot resisting sum off wonder pains thickness map barrows sea messenger won ankles."})
continuations250[18].push({selected : true, item : "Mixed_18", s : "that the singer who the fans idolized /won a prize /was exciting.", a : "x-x-x exists lose die occupy gas car sine ballpark sky sky alert dry illusion.", n : "CLAIM FACT"})
continuations250[18].push({selected : true, item : "Mixed_18", s : "that the singer who the fans idolized /made people happy /was exciting.", a : "x-x-x occurs wild joy induce lie ad heal unusable cent nation plate lot regiment."})
continuations250[19].push({selected : true, item : "Mixed_19", s : "that the CEO who the employee impressed /teased everyone /drove Bill crazy.", a : "x-x-x assume edge lot Opt big sin confined hereafter camped measured width Goal buyer.", n : "CLAIM FACT ACCUSATION"})
continuations250[19].push({selected : true, item : "Mixed_19", s : "that the CEO who the employee impressed /annoyed everyone /drove Bill crazy.", a : "x-x-x occurs debt ago Cam our us reaching tolerance unified sequence wheel Dear slept."})
continuations250[20].push({selected : false, item : "Mixed_20", s : "that the assistant who the philanthropist instructed /proved inept /was no surprise.", a : "x-x-x though dark sky prevented any map transcriptions deficiency muscle harps sit arm relating.", n : "CLAIM"})
continuations250[20].push({selected : false, item : "Mixed_20", s : "that the assistant who the philanthropist instructed /proved wrong /was no surprise.", a : "x-x-x assume thin nor requiring add sky collaborations aggressive beauty board air sit payments."})
continuations250[21].push({selected : false, item : "Mixed_21", s : "that the tycoon who the journalist interviewed /was evil /could not be disproven.", a : "x-x-x broken wild die teamed my ten invaluable restrictive sky task kinds thy law diacritic.", n : "CLAIM ACCUSATION"})
continuations250[21].push({selected : false, item : "Mixed_21", s : "that the tycoon who the journalist interviewed /was idiotic /could not be disproven.", a : "x-x-x hardly thin eat dammed non old usefulness stereotypes map inshore spite sun gas motocross."})
continuations250[22].push({selected : false, item : "Mixed_22", s : "that the boy who the bully intimidated /plagiarized his homework /dismayed his parents.", a : "x-x-x looked coal nor let net sky feats imperatives redeveloped mid shortest textures end require.", n : "CLAIM FACT ACCUSATION"})
continuations250[22].push({selected : false, item : "Mixed_22", s : "that the boy who the bully intimidated /drove everyone crazy /dismayed his parents.", a : "x-x-x decide sand lot so mid sky vomit homosexuals texts quantity crack emitting sex located."})
continuations250[23].push({selected : true, item : "Mixed_23", s : "that the sculptor who the painter invited /made sculptures /did not surprise anyone.", a : "x-x-x enough foot sin emphatic dry lie descent consist mind coronation sex arm earliest fourth.", n : "CLAIM FACT"})
continuations250[23].push({selected : true, item : "Mixed_23", s : "that the sculptor who the painter invited /made headlines /did not surprise anyone.", a : "x-x-x seeing gain nor secreted thy old dragged servant fact whichever sex joy sympathy survey."})
continuations250[24].push({selected : true, item : "Mixed_24", s : "that the principal who the teacher liked /turned everyone down /surprised the parents.", a : "x-x-x quoted mail eat regarding my lot neither pound annual tendency rest divisions lie suppose.", n : "CLAIM FACT"})
continuations250[24].push({selected : true, item : "Mixed_24", s : "that the principal who the teacher liked /calmed everyone down /surprised the parents.", a : "x-x-x seemed jobs eat therefore us joy exactly quick heyday carrying heat pollution sin develop."})
continuations250[25].push({selected : true, item : "Mixed_25", s : "that the dancer who the audience loved /was interviewed on TV /seemed exciting.", a : "x-x-x reduce wind fat ripped our how together peace gas magistrates sum Gap cotton monarchy.", n : "FACT CLAIM"})
continuations250[25].push({selected : true, item : "Mixed_25", s : "that the dancer who the audience loved /was featured on TV /seemed exciting.", a : "x-x-x sought warm sky canoes me law consists crime aid remnants end Ion estate treatise."})
continuations250[26].push({selected : true, item : "Mixed_26", s : "that the entrepreneur who the trickster misled /was admired everywhere /gave Mary fits.", a : "x-x-x remove anti lot completeness how box disguises devout sun caution newspapers book Post noon.", n : "CLAIM FACT"})
continuations250[26].push({selected : true, item : "Mixed_26", s : "that the entrepreneur who the trickster misled /was appreciated everywhere /gave Mary fits.", a : "x-x-x remove eyes fat interrelated big sin reprimand frenzy sin termination assumption jobs Poem ours."})
continuations250[27].push({selected : false, item : "Mixed_27", s : "that the musician who the father missed /injured the artist /came as a surprise.", a : "x-x-x unable pure thy differed red ten tissue eighty relaxed lie surely ways sat won velocity.", n : "CLAIM FACT ACCUSATION"})
continuations250[27].push({selected : false, item : "Mixed_27", s : "that the musician who the father missed /displeased the artist /came as a surprise.", a : "x-x-x unable weak die reproach sea non fairly breach unsettling sin resist flow sin sky villages."})
continuations250[28].push({selected : false, item : "Mixed_28", s : "that the babysitter who the parent paid /seemed competent /turned out to be unfounded.", a : "x-x-x argued warm ill pressuring thy me unable late notion sculpture window see arm ice statehood.", n : "CLAIM"})
continuations250[28].push({selected : false, item : "Mixed_28", s : "that the babysitter who the parent paid /seemed trustworthy /turned out to be unfounded.", a : "x-x-x filled hair sky doctorates why sea ensure fine notion adventurers tissue add ice thy pyramidal."})
continuations250[29].push({selected : true, item : "Mixed_29", s : "that the engineer who the tenant phoned /repaired the TV /was very believable.", a : "x-x-x broken cold nor allowing ill bad purity magnum esteemed fat Ha ill neck skyscraper.", n : "CLAIM FACT"})
continuations250[29].push({selected : true, item : "Mixed_29", s : "that the engineer who the tenant phoned /appeared on TV /was very believable.", a : "x-x-x ensure fear eat likewise sun ago sorrow smoker physical sex Ion eat mode randomness."})
continuations250[30].push({selected : false, item : "Mixed_30", s : "that the mobster who the media portrayed /seemed corrupt /gave Josh the chills.", a : "x-x-x taught wood ago offside big few meant ornaments damage crowned task Vest saw duplex.", n : "CLAIM ACCUSATION"})
continuations250[30].push({selected : false, item : "Mixed_30", s : "that the mobster who the media portrayed /seemed unreliable /gave Josh the chills.", a : "x-x-x merely warm thy orbited ice ice equal celestial visual postulated poem Jerk thy curate."})
continuations250[31].push({selected : true, item : "Mixed_31", s : "that the trickster who the witness recognized /was arrested /calmed people down.", a : "x-x-x taught acid joy converged joy sea quietly increasing big garrison angina return page.", n : "CLAIM FACT"})
continuations250[31].push({selected : true, item : "Mixed_31", s : "that the trickster who the witness recognized /was acknowledged /calmed people down.", a : "x-x-x argued heat ill nicknames why us realize everything joy differential italic health wine."})
continuations250[32].push({selected : false, item : "Mixed_32", s : "that the vendor who the storeowner recruited /welcomed the client /had excited the boss.", a : "x-x-x merely acid die aboard bad nor mothballed portraits inviting sky unable sin hunting map bark.", n : "CLAIM FACT"})
continuations250[32].push({selected : false, item : "Mixed_32", s : "that the vendor who the storeowner recruited /satisfied the client /had excited the boss.", a : "x-x-x become cold fat richer him ago tiebreaker flowering elections joy assume joy lesions run amid."})
continuations250[33].push({selected : true, item : "Mixed_33", s : "that the scientist who the mayor relied on /had faked data /made Sue angry.", a : "x-x-x quoted debt thy decreases lie fat worry studio buy mid evict rose ring Cow trend.", n : "CLAIM ACCUSATION"})
continuations250[33].push({selected : true, item : "Mixed_33", s : "that the scientist who the mayor relied on /couldn’t be trusted /made Sue angry.", a : "x-x-x anyone acid sin admirable fat big yours purity mid outtakes thy peasant drop Inn mills."})
continuations250[34].push({selected : true, item : "Mixed_34", s : "that the bureaucrat who the guard shouted at /instructed the newscaster /was bogus.", a : "x-x-x toward heat die redirected sea non males clauses bit judgments bed footbridge joy limbo.", n : "CLAIM"})
continuations250[34].push({selected : true, item : "Mixed_34", s : "that the bureaucrat who the guard shouted at /disturbed the newscaster /was bogus.", a : "x-x-x remove dark nor speculator red bad waves founder car judgments map outfielder mid sonar."})
continuations250[35].push({selected : true, item : "Mixed_35", s : "that the employee who the manager supervised /was reprimanded /devastated his family.", a : "x-x-x argued skin nor conclude sun sky another precursors eat sympathized stagnation sit filled.", n : "CLAIM FACT"})
continuations250[35].push({selected : true, item : "Mixed_35", s : "that the employee who the manager supervised /was dismissed /devastated his family.", a : "x-x-x inches sons thy overcome sky sin beneath embodiment fat precision childbirth eat heaven."})
continuations250[36].push({selected : true, item : "Mixed_36", s : "that the president who the commander supported /was impeached /shocked the people.", a : "x-x-x unable food fat recognize lie red obtaining evolution sun kinematic poultry put before.", n : "CLAIM FACT"})
continuations250[36].push({selected : true, item : "Mixed_36", s : "that the president who the commander supported /was confirmed /shocked the people.", a : "x-x-x quoted food joy therefore old how describes substance bed provinces catalog lie allows."})
continuations250[37].push({selected : false, item : "Mixed_37", s : "that the clerk who the customer talked to /was heroic /appeared to be a lie.", a : "x-x-x remove slow die solve mid old scarcely artist joy sun grains frequent bed thy thy her.", n : "CLAIM"})
continuations250[37].push({selected : false, item : "Mixed_37", s : "that the clerk who the customer talked to /was sad /appeared to be a lie.", a : "x-x-x occurs sand nor admit ice old weakness verbal fat sun ion advanced map sky nor it."})
continuations250[38].push({selected : true, item : "Mixed_38", s : "that the driver who the tourist thanked /was ill /bothered Bill.", a : "x-x-x indeed lose sin smiled old mid declare rushing ill joy postpone Port.", n : "CLAIM FACT"})
continuations250[38].push({selected : true, item : "Mixed_38", s : "that the driver who the tourist thanked /was crazy /bothered Bill.", a : "x-x-x argued fine ill enable me me despair tending sin quest escorted Base."})
continuations250[39].push({selected : true, item : "Mixed_39", s : "that the thief who the detective tracked /robbed everyone /was disconcerting.", a : "x-x-x taught mail die heels why six ascending caliber escort equation fat consequential.", n : "CLAIM FACT ACCUSATION"})
continuations250[39].push({selected : true, item : "Mixed_39", s : "that the thief who the detective tracked /frightened everyone /was disconcerting.", a : "x-x-x assume weak thy grams dry why adherents hospice memorandum equation war resuscitation."})
continuations250[40].push({selected : true, item : "Mixed_40", s : "that the athlete who the coach trained /expressed anger /came as a disappointment.", a : "x-x-x accept foot ago promos ill hot vivid illness centuries pound sale eat die ecclesiastical.", n : "CLAIM FACT ACCUSATION"})
continuations250[40].push({selected : true, item : "Mixed_40", s : "that the athlete who the coach trained /provoked anger /came as a disappointment.", a : "x-x-x caused slow joy combos mid bad shake tobacco stubborn souls book eye die qualifications."})
continuations250[41].push({selected : true, item : "Mixed_41", s : "that the runner who the psychiatrist treated /won the marathon /seemed very interesting.", a : "x-x-x remove feet ago vanish own sky recognizable correct sin thy executes cities poem distinction.", n : "CLAIM FACT"})
continuations250[41].push({selected : true, item : "Mixed_41", s : "that the runner who the psychiatrist treated /was widely known /seemed very interesting.", a : "x-x-x ensure real ill spears sky why orientations aspects lot sector least stages mail application."})
continuations250[42].push({selected : true, item : "Mixed_42", s : "that the doctor who the patient visited /was talented /impressed the whole city.", a : "x-x-x inches mail thy heaven sun sea towards finance joy flourish recommend sky alone lose.", n : "CLAIM"})
continuations250[42].push({selected : true, item : "Mixed_42", s : "that the doctor who the patient visited /was credible /impressed the whole city.", a : "x-x-x anyone wait sky inches net joy whether delight big romances declining sit heard cent."})
continuations250[43].push({selected : true, item : "Mixed_43", s : "that the pediatrician who the receptionist worked for /mistrusted the parent /had troubled people.", a : "x-x-x seeing coal sin accelerators joy joy mountainside vision map exonerated joy toward sky speeches moment.", n : "CLAIM FACT ACCUSATION"})
continuations250[43].push({selected : true, item : "Mixed_43", s : "that the pediatrician who the receptionist worked for /disturbed the parent /had troubled people.", a : "x-x-x caused lose fat exaggerating joy big invertebrate extent net numerical map decide sky garrison valley."})










// Annotate regions
for(i=0; i<continuations250.length; i++) {
	for(j=0; j<2; j++) {
		s = continuations250[i][j].s.split(" ");
		regions = [];
		reg_id = 0;
		within_reg = 0;
		for(k = 0; k<s.length; k++) {
			if(k == 3) {
				reg_id = 1;
				within_reg = 0;
			} else if(s[k][0] == "/") {
				reg_id ++;
				within_reg = 0;
  			       s[k] = s[k].substr(1);
			}
			regions.push("REGION_"+reg_id+"_"+within_reg);
			within_reg++;
		}
		continuations250[i][j].s = s.join(" ")
		continuations250[i][j].r = regions.join(" ");
		console.log(continuations250[i][j]);
		console.log(s.length, regions.length);
	}
}

console.log(continuations250);

continuations250_ = [];
for(i=0; i < continuations250.length; i++) {
	console.log(i);
	console.log(continuations250[i]);
	console.log(continuations250[i][0].exclude);
	if(continuations250[i][0].selected == true) {
		continuations250_.push(continuations250[i]);
	}
}
continuations250 = continuations250_;


continuations = continuations250;

console.log(continuations);



for(i = 0; i<continuations.length; i++) {
	console.log(i, continuations[i]);
	for(j =0; j<2; j++) {
                continuations[i][j].condition = "continuations"
	}
	console.log(continuations[i]);
	continuations[i][0].n = continuations[i][0].n.split(" ")
	continuations[i][1].n = continuations[i][0].n
}





// Shuffle the items
continuationsChosen = continuations.sort(() => Math.random() - 0.5); // https://javascript.info/task/shuffle
console.log(continuationsChosen);


// Now, for each selected noun find some matching item from the appropriate semantic class

console.log(topNouns);

FAILED = true;
matching_attempts = 0;
while(FAILED) {
topNouns = _.shuffle(topNouns);

matching_attempts = matching_attempts+1;
	if(matching_attempts > 10) { // if matching failed 10 times, give up. This should not happen, but is a precaution to prevent the script from taking up the participant's browser's resoures.
		CRASH();
	}
	nounsAndVerbsCopied = [...continuationsChosen]
        nounsAndVerbsAssignment = []
	FAILED = false;
//	console.log(topNouns);
        for(n = 0; n<topNouns.length; n++) {
//		console.log(n);
                relevantContinuations = [];
        	noun = topNouns[n]
//		console.log(noun);
                for(c = 0; c<nounsAndVerbsCopied.length; c++) {
//			console.log(c);
//			console.log(nounsAndVerbsCopied[c].n);

			for(d = 0; d<nounsAndVerbsCopied[c][0].n.length; d++) {
//				console.log(nounsAndVerbsCopied[c].n[d]);
//				console.log(noun, nouns[nounsAndVerbsCopied[c].n[d]]);
                                console.log(nounsAndVerbsCopied[c][0].n, d);
	                	if(nouns[nounsAndVerbsCopied[c][0].n[d]].includes(noun)) {
				//	console.log("found "+noun);
                        	   relevantContinuations.push(c);
                 	   	}
			}
                }
		console.log("CONSIDERING NOUN", noun, n, "Remaining conts.", nounsAndVerbsCopied.length, relevantContinuations);
                if(relevantContinuations.length == 0) {
                	FAILED = true;
                	break
                } else {
                       chosen = _.sample(relevantContinuations, 1);
			//console.log(chosen);
			nounsAndVerbsAssignment.push(nounsAndVerbsCopied[chosen]);
			nounsAndVerbsCopied.splice(chosen, 1);
			//console.log(nounsAndVerbsCopied.length)
                }
        }
        console.log("ATTEMPTS", matching_attempts);
}



console.log(nounsAndVerbsAssignment);
console.log(topNouns);
conditionAssignment = _.shuffle(conditionAssignment);

continuations = nounsAndVerbsAssignment;
console.log(conditionAssignment);

// Now assign conditions

conditions_chosen = [];
continuationsChosen = []
for(i = 0; i<continuations.length; i++) {
		continuations[i][0].noun = topNouns[i];
		continuations[i][1].noun = topNouns[i];
		continuations[i][0].s = "The "+topNouns[i]+" "+continuations[i][0].s;
		continuations[i][1].s = "The "+topNouns[i]+" "+continuations[i][1].s;
		continuations[i][0].r = "REGION_D0 REGION_N0 "+continuations[i][0].r;
		continuations[i][1].r = "REGION_D0 REGION_N0 "+continuations[i][1].r;
        console.log(continuations[i]);
	if(conditionAssignment[i] < 2) {
		item = continuations[i][0];
		if(conditionAssignment[i] == 0) { //0 == "critical"compatible"
    		    item["condition"] = "critical_incompatible"
		} else if(conditionAssignment[i] == 1) {
    		    item["condition"] = "critical_NoSC" // 1 == "critical_NoSC"
		} else if (conditionAssignment[i] == -1) { // -1 = critical_SCRC_incompatible
    		    item["condition"] = "critical_SCRC_incompatible"
		}

		distractors1 = continuations[i][0].a.split(" ");
		distractors2 = continuations[i][1].a.split(" ");
		words1 = continuations[i][0].s.split(" ")
		words2 = continuations[i][1].s.split(" ")
		distractors = [];
		regions = item.r.split(" ");
		item.distractor_condition = "none";
		for(j = 0; j<words2.length; j++) {
				distractors.push(distractors1[j]);
		}
		item.a = distractors.join(" ");
	} else {
		item = continuations[i][1];
		if(conditionAssignment[i] == 2) {
  		    item["condition"] = "critical_compatible" // 2 == "critical_compatible"
		} else if(conditionAssignment[i] == 3) {
			item["condition"] = "critical_SCRC_compatible" // 3 == "critical_SCRC_incompatible"
		} else {
			CRASH();
		}
		distractors1 = continuations[i][0].a.split(" ");
		distractors2 = continuations[i][1].a.split(" ");
		words1 = continuations[i][0].s.split(" ")
		words2 = continuations[i][1].s.split(" ")
		distractors = [];
		regions = item.r.split(" ");
		item.distractor_condition = "none";
		for(j = 0; j<words2.length; j++) {
			console.log(continuations[i][0]);
			if(j < regions.length && (words1[j] != words2[j]) || (j > 0 && words1[j-1] != words2[j-1])) { // if the word or the previous word differ between the conditions, use the special distractor. For the VERB-the-NOUN stimuli, this ensures that the distractor on the precritical word is the same across conditions
			// Note: This logic was coded for the compatibility manipulation, which was not used in this experiment.
				distractors.push(distractors2[j]);
			} else {
				distractors.push(distractors1[j]);
			}
		}
		item.a = distractors.join(" ");

	}

	s_ = [];
	a_ = [];
	r_ = [];
        s = item.s.split(" ")
	a = item.a.split(" ")
	r = item.r.split(" ")
	console.log(r);
	condition = item["condition"]

	for(j=0; j<s.length; j++) {
		if(condition == "critical_NoSC" && r[j].startsWith("REGION_0_")) {
			continue;
		}
		if(condition == "critical_NoSC" && r[j].startsWith("REGION_2_")) {
			continue;
		}
		if(condition != "critical_SCRC_compatible" && condition != "critical_SCRC_incompatible" && r[j].startsWith("REGION_1_")) {
			continue;
		}
		s_.push(s[j]);
		a_.push(a[j]);
		r_.push(r[j]);
	}
        item.s = s_.join(" ")
        item.a = a_.join(" ")
        item.r = r_.join(" ")


        continuationsChosen.push(item)
conditions_chosen.push(condition);
}


console.log("CONTINUATIONS CHOSEN");
console.log(continuationsChosen);
criticalChosen = continuationsChosen
console.log(conditions_chosen.sort());
exp.conditions_chosen = conditions_chosen.sort();





fillers = []
fillers.push({s:"The divorcee has come to love her life ever since she got divorced.", a:"x-x-x nearly else bed took fell lord cup air stand base web keyboard.", q : "? Does the divorcee feel positively towards her life? Y"}) 
fillers.push({s:"The mathematician at the banquet baffled the philosopher although she rarely needed anyone else's help.", a:"x-x-x rebelling trip lot corpses audible kept inspections appeared card branch moving happen polish oh.", q : "? Was the mathematician baffled by the philosopher? N"}) 
fillers.push({s:"The showman travels to different cities every month.", a:"x-x-x citing terrain hall certainly listen write rates.", q : "? Does the showman never travel? N"}) 
fillers.push({s:"The roommate takes out the garbage every week.", a:"x-x-x attest doubt sold lose enables worst anti.", q : "? Is the garbage taken out every day? N"}) 
fillers.push({s:"The dragon wounded the knight although he was far too crippled to protect the princess.", a:"x-x-x endorses funding plan borrow question walk tree pop key teammate stay society map indicate.", q : "? Did the dragon wound the knight? Y"}) 
fillers.push({s:"The office-worker worked through the stack of files on his desk quickly.", a:"x-x-x appreciating forget arrived lady prone wife treat fall born rain western.", q : "? Did the office-worker ignore the stack of files on his desk? N"}) 
fillers.push({s:"The firemen at the scene apprehended the arsonist because there was a great deal of evidence pointing to his guilt.", a:"x-x-x originate war sure among outsourcing cent deviance anymore mouth fun us enter laws yes produced observer plus bill weigh.", q : "? Was there a great deal of evidence pointing to the arsonist's guilt? Y"}) 
fillers.push({s:"During the season, the choir holds rehearsals in the church regularly.", a:"x-x-x nice called, us haunt anger prophecies laws thus issues customers.", q : "? Does the choir rehearse regularly during the season? Y"}) 
fillers.push({s:"The speaker who the historian offended kicked a chair after the talk was over and everyone had left the room.", a:"x-x-x criticises holy sad activated fraction upside mom files cases lot know port lord holy products port van guy how.", q : "? Had everyone to stay in the room after the talk was over? N"}) 
fillers.push({s:"The milkman punctually delivers the milk at the door every day.", a:"x-x-x obstruct clerestory lesbians lose quit ass nor took weird join.", q : "? Is the milkman on time every day? Y"}) 
fillers.push({s:"The quarterback dated the cheerleader although this hurt her reputation around school.", a:"x-x-x empties fairy sit propagation violence tell east lake represents access placed.", q : "? Did the date hurt the reputation of the cheerleader? Y"}) 
fillers.push({s:"The citizens of France eat oysters.", a:"x-x-x allege anti Amount girl lattice.", q : "? Do the French eat oysters? Y"}) 
fillers.push({s:"The bully punched the kid after all the kids had to leave to go to class.", a:"x-x-x arousing rituals eat what birth felt ha ha sun lake forms link jack size feels.", q : "? Did the bully punch the kid before all the kids had to leave? N"}) 
fillers.push({s:"After the argument, the husband ignored his wife.", a:"x-x-x plus suggests, cent without harmony seen here.", q : "? Was the wife being ignored by her husband after the argument? Y"}) 
fillers.push({s:"The engineer who the lawyer who was by the elevator scolded blamed the secretary but nobody listened to his complaints.", a:"x-x-x succumbing oh ha defend feet mine ones ha shouting rescind ounces sort including ass happen infantry laws far protecting.", q : "? Did the engineer blame the secretary? Y"}) 
fillers.push({s:"The librarian put the book onto the shelf.", a:"x-x-x impede east grow this wave grow bacon.", q : "? Did the clerk put the book on the shelf? N"}) 
fillers.push({s:"The photographer processed the film on time.", a:"x-x-x prematurely eliminate ago yes non nor.", q : "? Was the photographer late? N"})
fillers.push({s:"The spider that the boy who was in the yard captured scared the dog since it was larger than the average spider.", a:"x-x-x enclosing sad cent been hell pro say jack earn resource expert file gets ended list per decide lady anti imagine quotes.", q : "? Was the spider scared by the dog? N"}) 
fillers.push({s:"The sportsman goes jogging in the park regularly.", a:"x-x-x incurring hear outback hope fell been processes.", q : "? Does the sportsman jog in the forest? N"}) 
fillers.push({s:"The customer who was on the phone contacted the operator because the new long-distance pricing plan was extremely inconvenient.", a:"x-x-x equates okay yeah bill sun maybe desperate wish wondered married link an unfortunately chronic miss yes residence inscriptions.", q : "? Was the new long-distance pricing plan convenient? N"}) 
fillers.push({s:"The private tutor explained the assignment carefully.", a:"x-x-x reproduce bumps amendment lot kilometers centuries.", q : "? Was the private tutor able to explain the assignment carefully? Y"}) 
fillers.push({s:"The audience who was at the club booed the singer before the owner of the bar could remove him from the stage.", a:"x-x-x solidly anti mid sir why me levee glad argued larger rich lying east done yes worse allows term file rose there.", q : "? Did the owner of the bar remove the singer from the stage after the audience had booed him? Y"}) 
fillers.push({s:"The defender is constantly scolding the keeper.", a:"x-x-x disembark sick definition dilation yeah albeit.", q : "? Has anybody been scolded? Y"}) 
fillers.push({s:"The hippies who the police at the concert arrested complained to the officials while the last act was going on stage.", a:"x-x-x possesses sale room anyone oh fit writers resource completion kill cup discussed worst damn yes grow sick worry sir older.", q : "? Did the singers complain to the officials? N"}) 
fillers.push({s:"The natives on the island captured the anthropologist because she had information that could help the tribe.", a:"x-x-x emanating fat else forget managers plan misconceptions release pick away combination die gonna damn gets shake.", q : "? Did the anthropologist have information that could destroy the tribe? N"}) 
fillers.push({s:"The trainee knew that the task which the director had set for him was impossible to finish within a week.", a:"x-x-x recursively easy jack eat earn prime note together wind word lose anti girl commission gun served degree cup thus.", q : "? Did the director set a task for the trainee? Y"}) 
fillers.push({s:"The administrator who the nurse from the clinic supervised scolded the medic while a patient was brought into the emergency room.", a:"x-x-x unmask hell fact forth none anti scales detectives pungent nice smoky match lake islands boys imagine view luck recommend able.", q : "? Was the administrator supervised by the nurse? Y"}) 
fillers.push({s:"The company was sure that its new product, which its researchers had developed, would soon be sold out.", a:"x-x-x closely mind dad sir cent nor another, throw drug accompanied eyes everybody, south page ha trip whom.", q : "? Was the company optimistic about the new product? Y"}) 
fillers.push({s:"The astronaut that the journalists who were at the launch worshipped criticized the administrators after he discovered a potential leak in the fuel tank.", a:"x-x-x supervises oh oh necessarily bed sure size yeah hungry vigorously calculated died reinforcements gotta rose electrical kept countries dean pain told laid cat.", q : "? Did the astronaut discover a potential leak in the fuel tank? N"}) 
fillers.push({s:"The janitor who the doorman who was at the hotel chatted with bothered a guest but the manager decided not to fire him for it.", a:"x-x-x conclude fat us intakes east ones miss ha today bedding mid tendency vote woods oh law however healthy rest kid wide road lake jack.", q : "? Did the manager decide not to fire the doorman? Y"}) 
fillers.push({s:"The technician at the show repaired the robot while people were taking a break for coffee.", a:"x-x-x devoting hate been guys comrades cup sells sweet stupid sale policy met today sale cannot.", q : "? Did the technician who was in the laboratory repair the robot? N"}) 
fillers.push({s:"The salesman feared that the printer which the customer bought was damaged.", a:"x-x-x dosing robust walk bar knocked weeks mid sciences impact map premier.", q : "? Was the salesman sure that the printer would work? N"}) 
fillers.push({s:"The students studied the surgeon whenever he performed an important operation.", a:"x-x-x reused summary stay advised indicate file something cent president companies.", q : "? Was the surgeon observed whenever he performed an important operation? Y"}) 
fillers.push({s:"The locksmith can crack the safe easily.", a:"x-x-x exert okay firms met took agreed.", q : "? Is it easy for the locksmith to crack the safe? Y"}) 
fillers.push({s:"The woman who was in the apartment hired the plumber despite the fact that he couldn't fix the toilet.", a:"x-x-x seeking cool sea hear ass basically plain lie jerseys reached eyes came mom sit football bell cent enters.", q : "? Was the plumber hired by the women at the office? N"}) 
fillers.push({s:"Yesterday the swimmer saw only a turtle at the beach.", a:"x-x-x nice hurdles ways fund web intake anti sold china.", q : "? Did the swimmer see a whale at the beach yesterday? N"}) 
fillers.push({s:"The surgeon who the detective who was on the case consulted questioned the coroner because the markings on the body were difficult to explain.", a:"x-x-x responding way web belonging bad girl ways soul hope databases profitable soul bullion playing hour explores ball won fun hope statement town windows.", q : "? Was the surgeon consulted by the detective who was on the case? Y"}) 
fillers.push({s:"The gangster who the detective at the club followed implicated the waitress because the police suspected he had murdered the shopkeeper.", a:"x-x-x rejoining lack how arbitrary far came held economic contracted park realizes animals read except religions bed case displays size furthering.", q : "? Was the waitress implicated by the gangster? Y"}) 
fillers.push({s:"During the party everybody was dancing to rock music.", a:"x-x-x buy comes otherwise few monster pay ago agree.", q : "? Were all the people just sitting around chatting at the party? N"}) 
fillers.push({s:"The fans at the concert loved the guitarist because he played with so much energy.", a:"x-x-x besting holy via citizen older seat cooperate limited keep cancer sit does mass months.", q : "? Was the guitarist loved by the fans? Y"}) 
fillers.push({s:"The intern comforted the patient because he was in great pain.", a:"x-x-x predate receptive wind noticed percent kid move park basis win.", q : "? Did someone comfort the patient? Y"}) 
fillers.push({s:"The casino hired the daredevil because he was confident that everything would go according to plan.", a:"x-x-x commences sword yes universes protect does her describes add understand china six authority ways down.", q : "? Was the daredevil worried about the plan? N"}) 
fillers.push({s:"The beggar is often scrounging for cigarettes.", a:"x-x-x officially mid feels concourses fan agreements.", q : "? Does the beggar always buy cigarettes? N"}) 
fillers.push({s:"The cartoonist who the readers supported pressured the dean because she thought that censorship was never appropriate.", a:"x-x-x diversifying heat god whoever communist legalized jack den perfect keep account oh affiliates feet learn description.", q : "? Did the authors pressure the dean? N"}) 
fillers.push({s:"The prisoner who the guard attacked tackled the warden although he had no intention of trying to escape.", a:"x-x-x certainly luck fine aimed suitable teaming mind invent congress mom grow boy describes pick author walk poetry.", q : "? Was the prisoner attacked by the guard although he was not trying to escape? Y"}) 
fillers.push({s:"The passer-by threw the cardboard box into the trash-can with great force.", a:"x-x-x succumbs quiet draw equitable his lord wish quarterly born agree agree.", q : "? Did the passer-by put the cardboard box into his bag? N"}) 
fillers.push({s:"The biker who the police arrested ran a light since he was driving under the influence of alcohol.", a:"x-x-x rehabilitate risk glad except breaking pain goal exist reach till loss opinion rules nor presented find discuss.", q : "? Did the police arrest the car driver? N"}) 
fillers.push({s:"The scientists who were in the lab studied the alien while the blood sample was run through the computer.", a:"x-x-x evict holy yes add goes bob monster son lacks wanna lie agree update wish ha reality note everyone.", q : "? Did the scientists take pictures of the alien? N"}) 
fillers.push({s:"The student quickly finished his homework assignments.", a:"x-x-x putting healthy southern wife airports magistrates.", q : "? Have the homework assignments been finished by the student? Y"}) 
fillers.push({s:"The environmentalist who the demonstrators at the rally supported calmed the crowd until security came and sent everyone home.", a:"x-x-x angering yeah sad perpendicular bed lot valve marketing spills best laugh spend contract me sure mom function hair.", q : "? Was everyone sent to prison by the security? N"}) 
fillers.push({s:"The producer shoots a new movie every year.", a:"x-x-x shortly pierce page anti enjoy peace mom.", q : "? Does the director produce a new movie every year? N"}) 
fillers.push({s:"The rebels who were in the jungle captured the diplomat after they threatened to kill his family for not complying with their demands.", a:"x-x-x memorably girl body soul girl visits memories card nuisance feels guys scientists says able move please pain ball nostalgic sir learn drivers.", q : "? Did the rebels threaten to kill the family of the diplomat? Y"}) 
fillers.push({s:"Dinosaurs ate other reptiles during the stone age.", a:"x-x-x earl write exporter minute guys wants dad.", q : "? Were the dinosaurs carnivores? Y"}) 
fillers.push({s:"The manager who the baker loathed spoke to the new pastry chef because he had instituted a new dress code for all employees.", a:"x-x-x contemplates anti map walks tenuous voted ass goal anti devoid skip weekend star mind veterinary lose dad sides want rose knew indicates.", q : "? Did the manager speak to the new pastry chef because of the dress code? Y"}) 
fillers.push({s:"The teacher doubted that the test that had taken him a long time to design would be easy to answer.", a:"x-x-x totalling grinder star feet them your miss miles song anti oh her ha posted enjoy door fund foot county.", q : "? Did the teacher design the test quickly? N"}) 
fillers.push({s:"The cook who the servant in the kitchen hired offended the butler and then left the mansion early to see a movie at the local theater.", a:"x-x-x admirably trip cell justify cool lose wanting rough collapse runs thirds gold term miss rate evolved ideas bill code mean miles yeah hear their acquire.", q : "? Did the servant in the kitchen hire the cook? Y"}) 

for(i=0; i<fillers.length; i++) {
	fillers[i].condition = "filler"
	fillers[i].item = "Filler_"+i
}

	practice = [];

practice.push({s:"The semester will start next week, but the students and teachers are not ready.", a:"x-x-x thrives anti wages body sold, sin sky entitled sky concrete oil him goods.", q : "? Are the teachers ready? N", practice : true})
practice.push({s:"The mother of the prisoner sent him packages that contained cookies and novels.", a:"x-x-x defraud dry arm amounted rare nor rhythmic fund authority blossom me defect.", q : "? Did the mother send packages? Y", practice : true})
practice.push({s:"The reporter had dinner yesterday with the baseball player who Kevin admired.", a:"x-x-x quantify joy reduce organisms rise sum attained tended sin Troop flowing.", q : "? Did the reporter speak with a football player? N", practice : true})
practice.push({s:"The therapist set up a meeting with the upset woman and her husband yesterday.", a:"x-x-x forestall ten sit sum absence wave ran keeps exist dry sum settled remainder.", q : "? Was the woman upset? Y", practice : true})

for(i=0; i<practice.length; i++) {
	practice[i].condition = "filler"
	practice[i].item = "Practice_"+i
}




function separatedShuffle(x, y) {
	indices_x = [...Array(x.length).keys()].map(function(x){ return ["x",x]})
	indices_y = [...Array(y.length).keys()].map(function(x){ return ["y",x]})
	if(indices_x.length <= indices_y.length+5) {
		CRASH()
	}
	console.log(indices_x);
	console.log(indices_y);
	// shuffle
	indices_y = _.shuffle(indices_y);

	result = indices_x.concat(indices_y.slice(1, indices_y.length));
	attempts_order = 0;
	console.log("SHUFFLING");
	result = _.shuffle(result);
	// now add the remaining critical trial at the end
	result.push(indices_y[0]);
	for(i=0; i+1<result.length; i++) {
		if(result[i][0] == "y" && result[i+1][0] == "y") {
			candidate_positions = [];
	                for(j=0; j+3<result.length; j++) {
                           if(result[j][0] == "x" && result[j+1][0] == "x" && result[j+2][0] == "x" && result[j+3][0] == "x") {
				   candidate_positions.push(j+2);
			   }
			}
			console.log(i, "CANDIDATES", candidate_positions);
			SELECTED_NEW_POSITION = _.sample(candidate_positions, 1)[0];
			X = result[i];
			Y = result[SELECTED_NEW_POSITION]
			result[SELECTED_NEW_POSITION] = X;
			result[i] = Y;
		}
		if(i+2<result.length && result[i][0] == "y" && result[i+1][0] == "x" && result[i+2][0] == "y") {
			candidate_positions = [];
	                for(j=0; j+3<result.length; j++) {
                           if(result[j][0] == "x" && result[j+1][0] == "x" && result[j+2][0] == "x" && result[j+3][0] == "x") {
				   candidate_positions.push(j+2);
			   }
			}
			console.log(i, "CANDIDATES", candidate_positions);
			SELECTED_NEW_POSITION = _.sample(candidate_positions, 1)[0];
			X = result[i];
			Y = result[SELECTED_NEW_POSITION]
			result[SELECTED_NEW_POSITION] = X;
			result[i] = Y;
		}
	}
	for(i=0; i+1<result.length; i++) {
		if(result[i][0] == "y" && result[i+1][0] == "y") {
			console.log("THIS SHOULD NOT HAPPEN", i);
		}
	}
	result_ = []
	for(i = 0; i<result.length; i++) {
		if(result[i][0] == "x") {
			result_.push(x[result[i][1]]);
		} else {
			result_.push(y[result[i][1]]);
		}
	}
	return result_;
}

console.log("CRITICAL", criticalChosen);

     fillersAndCritical = separatedShuffle(_.sample(fillers, 30), criticalChosen);
//     fillersAndCritical = criticalChosen;



     fullStimuli = _.shuffle(practice).concat(fillersAndCritical);


item_ids = [];
//for(i = 0; i<fullStimuli.length; i++) {
//	console.log("PARTITIONED", i);
//	console.log(i, fullStimuli[i].item);
//}
//


     console.log( fullStimuli);
     return fullStimuli;
     
}

