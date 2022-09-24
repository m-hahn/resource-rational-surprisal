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

exp.condition = 0; //Math.floor(Math.random()*2);
console.log("Condition "+exp.condition)





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


     fillers = [];
     stims = [];
	critical = [];
     
     conditionAssignment = [];
     conditionsCounts = [0, 0, 0, 0];
	if(Math.random() > 0.5) {
		cond1 = 3;
		cond2 = -1;
		cond3 = 0;
		cond4 = 2;
	} else {
		cond1 = 0;
		cond2 = 2;
		cond3 = 3;
		cond4 = -1;
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


//topNouns = [];

console.log("------------------------")
console.log("TOP NOUNS BEFORE SLICING", topNouns);

topNouns1 = _.sample(topNouns.slice(0, 15), 10);
topNouns2 = _.sample(topNouns.slice(topNouns.length-15, topNouns.length), 10);
//topNouns2 = topNouns.slice(10, 20);


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


// for now, only use topNouns1


console.log("LENGTH", topNouns.length, topNouns);



continuations238 = []

for(i=0; i<18; i++) {
	continuations238.push([]);
}


continuations238[0].push({item : "238_Critical_VN1", s : "that the carpenter who the craftsman carried /confused the apprentice /was acknowledged.",a : "x-x-x killed knew sat reconcile dry add prompting economy decrease arm visitation net unemployment."   , n : "CLAIM FACT ACCUSATION"})
continuations238[0].push({item : "238_Critical_VN1", s : "that the carpenter who the craftsman carried /hurt the apprentice /was acknowledged.",a : "x-x-x broken stay add contended net non exponents highest oral mid adjectives net transmission."    , n : "CLAIM FACT ACCUSATION"})

continuations238[1].push({item : "238_Critical_VN2", s : "that the daughter who the sister supervised /frightened the grandmother /seemed concerning.",a : "x-x-x merely feel sat answered net sea inches detectives bargaining mid corresponds accept leadership."  , n : "CLAIM FACT ACCUSATION"})
continuations238[1].push({item : "238_Critical_VN2", s : "that the daughter who the sister supervised /greeted the grandmother /seemed concerning.",a : "x-x-x exists soft sat remember mid lot partly detectives sulfate arm compression accept connection."  , n : "CLAIM FACT ACCUSATION"})

continuations238[2].push({item : "238_Critical_VN3", s : "that the tenant who the foreman looked for /annoyed the shepherd /proved to be made up.",a : "x-x-x tissue acid sat assign net thy allergy sample sky integer fat inferred useful per was tone sat.", n : "CLAIM ACCUSATION"})
continuations238[2].push({item : "238_Critical_VN3", s : "that the tenant who the foreman looked for /questioned the shepherd /proved to be made up.",a : "x-x-x merely duty sat paused bad thy topical secure joy irrigation sum switched useful sea non item sea."  , n : "CLAIM ACCUSATION"})

continuations238[3].push({exclude : true, item : "238_Critical_VN4", s : "that the musician who the father missed /displeased the artist /confused the banker.",a : "x-x-x killed sand sat speedily thy fat enough immune imaginable sun varies meetings bed summed."    , n : "CLAIM FACT"})
continuations238[3].push({exclude : true, item : "238_Critical_VN4", s : "that the musician who the father missed /injured the artist /confused the banker.",a : "x-x-x walked side sat examines thy oil toward charts amazing mid prefer imperial fat spared." , n : "CLAIM FACT"})
continuations238[4].push({item : "238_Critical_VN5", s : "that the pharmacist who the stranger intimidated /distracted the customer /sounded surprising.",a : "x-x-x issued find sat allocating non map outlined dignitaries advocating sum wherever earthly complexity.", n : "CLAIM FACT", q : "? Did the NOUN distract the customer? @", q2 : "? Did the stranger see the pharmacist? @"})
continuations238[4].push({item : "238_Critical_VN5", s : "that the pharmacist who the stranger intimidated /questioned the customer /sounded surprising.",a : "x-x-x seemed rare sat discarding non map literacy dignitaries autonomous map scarcely earthly enthusiasm." , n : "CLAIM FACT", q : "? Did the NOUN sound surprising? @" })
continuations238[5].push({exclude : true, item : "238_Critical_VN6", s : "that the bureaucrat who the guard shouted at /disturbed the newscaster /annoyed the neighbor.",a : "x-x-x merely port sat overflowed big per admit phrases age catalogue run polyhedron infancy aid organize."  , n : "CLAIM FACT ACCUSATION", q : "? Did the NOUN disturb the newscaster? @" , q2 : "? Did the guard shout at the bureaucrat? @"})
continuations238[5].push({exclude : true, item : "238_Critical_VN6", s : "that the bureaucrat who the guard shouted at /instructed the newscaster /annoyed the neighbor.",a : "x-x-x remove mere sat alternated dry net shook impulse sky vegetables bar crossovers debated sin reviewed." , n : "CLAIM FACT ACCUSATION", q : "? Did the NOUN annoy the neighbor? @" })
continuations238[6].push({exclude : true, item : "238_Critical_VN7", s : "that the cousin who the brother described /troubled the uncle /startled the mother.",a : "x-x-x sought skin mid convey net non percent variation grateful net owned repaired ago enough."                , n : "CLAIM FACT ACCUSATION", q : "? Did the NOUN trouble the uncle? @", q2 : "? Did the brother describe the cousin? @"})
continuations238[6].push({exclude : true, item : "238_Critical_VN7", s : "that the cousin who the brother described /killed the uncle /startled the mother.",a : "x-x-x ensure wine sat detect me sea towards dangerous rings mid shook dissolve arm proved." , n : "CLAIM FACT ACCUSATION", q : "? Did the NOUN startle the mother? @"})

// TODO this is an error, erroenously having two 'item' entries.
continuations238[7].push({item : "238_Critical_VN8",  item : "Critical_6", s : "that the surgeon who the patient thanked /shocked his colleagues /was ridiculous.", a : "x-x-x assume pass sat notably us son suggest diocese thanked bar translated non containers.", n : "CLAIM"})
continuations238[7].push({item : "238_Critical_VN8",  item : "Critical_6", s : "that the surgeon who the patient thanked /cured his colleagues /was ridiculous.", a : "x-x-x looked milk sat erected us sum because scandal maxim arm newspapers non chromosome." , n : "CLAIM"})

continuations238[8].push({ item : "238_Critical_Vadv1", s : "that the commander who the president appointed /was confirmed yesterday /troubled people.", a : "x-x-x unable cold eat depending joy map evolution evolution big requiring parameter swimming scheme.", n : "CLAIM FACT"})
continuations238[8].push({ item : "238_Critical_Vadv1", s : "that the commander who the president appointed /was fired yesterday /troubled people.", a : "x-x-x exists skin ago evidently aid dry according challenge box tubes residence swimming begins.", n : "CLAIM FACT"})

continuations238[9].push({ item : "238_Critical_Vadv2", s : "that the trickster who the woman recognized /was acknowledged by the police /calmed people down.", a : "x-x-x prices food sin dissonant why non shown industries mid publications am ran across horned height arts.", n : "CLAIM FACT"})
continuations238[9].push({ item : "238_Critical_Vadv2", s : "that the trickster who the woman recognized /was arrested by the police /calmed people down.", a : "x-x-x itself skin sky converges joy son apply situations sky molecule net sit though horned shared zero.", n : "CLAIM FACT"})

continuations238[10].push({ item : "238_Critical_Vadv3", s : "that the politician who the farmer trusted /was refuted three days ago /did not bother the farmer.", a : "x-x-x looked iron sin resembling bad son dreams specify big rampart build soil ask bed ran heresy met plenty.", n : "CLAIM FACT"})
continuations238[10].push({ item : "238_Critical_Vadv3", s : "that the politician who the farmer trusted /was elected three days ago /did not bother the farmer.", a : "x-x-x trying wild ago introduces mid why whilst infants go windows plate tone add bed sky merged thy waited.", n : "CLAIM FACT"})

continuations238[11].push({exclude : true, item : "238_Critical_Vadv4", s : "that the dancer who the audience loved /was reported on TV /seemed very interesting.", a : "x-x-x tissue eyes ill octave big fat whenever ships joy contents dry Ion nature laws recognition.", n : "CLAIM FACT"})
continuations238[11].push({exclude : true, item : "238_Critical_Vadv4", s : "that the dancer who the audience loved /was interviewed on TV /seemed very interesting.", a : "x-x-x become wood thy repaid gas few whenever ships end affirmation am Hat nature sand consequence.", n : "CLAIM FACT"})

continuations238[12].push({ item : "238_Critical_Vadv5", s : "that the politician who the banker bribed /seemed credible to everyone /gave Josh the chills.", a : "x-x-x decide dark ill profoundly mid him relies fealty narrow meridian run contents tone Lain sky calmed.", n : "CLAIM FACT"})
continuations238[12].push({ item : "238_Critical_Vadv5", s : "that the politician who the banker bribed /seemed corrupt to everyone /gave Josh the chills.", a : "x-x-x though warm sky peculiarly dry dog latent ermine output inhibit ad examples tone Urea sin booths.", n : "CLAIM FACT"})

continuations238[13].push({ item : "238_Critical_Vadv6", s : "that the sculptor who the painter admired /made headlines in the US /did not surprise anyone.", a : "x-x-x sought coal eat retorted non per sincere leisure move laterally mid buy Sit joy sex inclined parent.", n : "CLAIM FACT"})
continuations238[13].push({ item : "238_Critical_Vadv6", s : "that the sculptor who the painter admired /made sculptures in the US /did not surprise anyone.", a : "x-x-x trying foot fat diffused how man barrier mercury safe undermined sky die Sit joy age resolved moment.", n : "CLAIM FACT"})

continuations238[14].push({ item : "238_Critical_Vadv7", s : "that the runner who the psychiatrist treated /was widely known in France /turned out to be incorrect.", a : "x-x-x filled dear lot infect ill ago cooperatives perhaps few latter rapid sat Valley entire per sin due ingenuity.", n : "CLAIM"})
continuations238[14].push({ item : "238_Critical_Vadv7", s : "that the runner who the psychiatrist treated /won the marathon in France /turned out to be incorrect.", a : "x-x-x slowly root ago unison sky aid inflationary reality sky sun isotopic thy Obtain entire own ad if dominions.", n : "CLAIM"})

continuations238[15].push({ item : "238_Critical_Vadv8", s : "that the analyst who the banker trusted /appeared on TV this morning /was very believable.", a : "x-x-x reduce boat sin retains dry fat prayed triumph patterns eat Ion ring suppose fat port evaporates.", n : "CLAIM"})
continuations238[15].push({ item : "238_Critical_Vadv8", s : "that the analyst who the banker trusted /repaired the TV this morning /was very believable.", a : "x-x-x reduce rise sky expects us gas closes glucose imminent map Ye rare degrees fat neck upholstery.", n : "CLAIM"})

continuations238[16].push({ item : "238_Critical_Vadv9", s : "that the child who the medic rescued /was mentioned in newspapers /seemed very interesting.", a : "x-x-x expect thou lot their sky ad stabs arsenal add mountains try discipline divine poet corporation.", n : "CLAIM FACT"})
continuations238[16].push({ item : "238_Critical_Vadv9", s : "that the child who the medic rescued /wrote articles in newspapers /seemed very interesting.", a : "x-x-x prices gain thy where us sky mocks bizarre curve concrete bar complexity divine knew governments.", n : "CLAIM FACT"})

continuations238[17].push({exclude : true,  item : "238_Critical_Vadv10", s : "that the CEO who the employee impressed /was annoying to Anne /dismayed Jeremy.", a : "x-x-x quoted iron sin BOP thy bad peculiar peninsula bit shortest act Flag foothold Rabbis.", n : "CLAIM FACT ACCUSATION"})
continuations238[17].push({exclude : true,  item : "238_Critical_Vadv10", s : "that the CEO who the employee impressed /was lying to Anne /dismayed Jeremy.", a : "x-x-x occurs fine fat JAB why bad wherever selecting mid crops die Seed dedicate Fronts.", n : "CLAIM FACT ACCUSATION"})



// TODO 238 Note the order the compatible and incompatible is wrong. For consistency, this is fixed downstream in the processing script. Later, better fix this here.


for(i=0; i<18; i++) {
	for(j=0; j<2; j++) {
		s = continuations238[i][j].s.split(" ");
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
		continuations238[i][j].s = s.join(" ")
		continuations238[i][j].r = regions.join(" ");
		console.log(continuations238[i][j]);
		console.log(s.length, regions.length);
	}
	a = continuations238[i][0];
	b = continuations238[i][1];
	continuations238[i] = [b,a];

}


console.log(continuations238);


continuations232 = []

for(i=0; i<32; i++) {
	continuations232.push([]);
}



continuations232[0].push({ item : "232_Critical_0", s : "that the teacher who the principal liked failed the student was only a malicious smear.", a : "x-x-x prices rare thy achieve fat bad certainly habit winter fat suppose non zone sun pronounce messy."     , n : "CLAIM ACCUSATION"})
continuations232[0].push({ item : "232_Critical_0", s : "that the teacher who the principal liked annoyed the student was only a malicious smear.", a : "x-x-x unless cash sin opposed joy son elsewhere proud fulfill sum thereby non goal am allusions vomit."     , n : "CLAIM ACCUSATION"})
continuations232[1].push({ item : "232_Critical_1", s : "that the doctor who the colleague mistrusted cured the patients seemed hard to believe.", a : "x-x-x tissue soft ago decide lot son constants dismissals axial map yourself motion ring thy therapy."     , n : "CLAIM FACT"})
continuations232[1].push({ item : "232_Critical_1", s : "that the doctor who the colleague mistrusted bothered the patients seemed hard to believe.", a : "x-x-x slowly salt sky assume oil red adjusting quadrupled cocktail day together motion port law highest."     , n : "CLAIM FACT"})
continuations232[2].push({ item : "232_Critical_2", s : "that the bully who the children hated harassed the boy was entirely correct.", a : "x-x-x unable salt sky cobra us bad yourself seats nuisance box am ice equation whereas." , n : "CLAIM FACT ACCUSATION"})
continuations232[2].push({ item : "232_Critical_2", s : "that the bully who the children hated shocked the boy was entirely correct.", a : "x-x-x toward male fat spade add dry involves super factual oil her ice agencies service." , n : "CLAIM FACT ACCUSATION"})
continuations232[3].push({ item : "232_Critical_3", s : "that the agent who the FBI sent arrested the criminal was acknowledged.", a : "x-x-x inches duty fat their joy ice Vie none locality eat becoming dry interactions." , n : "FACT CLAIM"})
continuations232[3].push({ item : "232_Critical_3", s : "that the agent who the FBI sent confused the criminal was acknowledged.", a : "x-x-x though rare sin since fat low Vie hour decrease mid everyone dry interactions." , n : "FACT CLAIM"})
continuations232[4].push({ item : "232_Critical_4", s : "that the senator who the diplomat supported defeated the opponent deserved attention.", a : "x-x-x placed jobs fat enhance fat hot porosity something downward sun evaluate obsolete mechanism." , n : "FACT CLAIM"})
continuations232[4].push({ item : "232_Critical_4", s : "that the senator who the diplomat supported troubled the opponent deserved attention.", a : "x-x-x become wood sky payable sea mid unfolded emotional reliance add preached obsolete afternoon." , n : "FACT CLAIM"})
continuations232[5].push({ item : "232_Critical_5", s : "that the fianc&eacute; who the author met married the bride did not surprise anyone.", a : "x-x-x please coal fat promos sin lot beyond red product map seeks aid ad entering effort." , n : "FACT CLAIM"})
continuations232[5].push({ item : "232_Critical_5", s : "that the fianc&eacute; who the author met startled the bride did not surprise anyone.", a : "x-x-x exists hear joy promos non son enough oil residues tax borne aid non pleasant factor." , n : "FACT CLAIM"})
continuations232[6].push({ item : "232_Critical_6", s : "that the businessman who the sponsor backed fired the employee came as a disappointment.", a : "x-x-x enough wine nor pedagogical joy ill reminds causal beans sin darkness soul aid lot specifications."     , n : "FACT CLAIM ACCUSATION"})
continuations232[6].push({ item : "232_Critical_6", s : "that the businessman who the sponsor backed surprised the employee came as a disappointment.", a : "x-x-x become ways nor nourishment few hot tyranny rulers varieties sin overcome soul eat sun ecclesiastical."     , n : "FACT CLAIM ACCUSATION"})
continuations232[7].push({ item : "232_Critical_7", s :"that the thief who the detective caught robbed the woman broke her family's heart.", a : "x-x-x merely acid nor stare ill red safeguard toward garage try those virus bit telethon these."     , n : "FACT CLAIM ACCUSATION"})
continuations232[7].push({ item : "232_Critical_7", s : "that the thief who the detective caught enraged the woman broke her family's heart.", a : "x-x-x unless evil lot steal sea sin emissions global lyrical joy shall virus am telethon speak."     , n : "FACT CLAIM ACCUSATION"})
continuations232[8].push({ item : "232_Critical_8", s : "that the criminal who the stranger distracted abducted the officer seemed concerning.", a : "x-x-x enough coal joy becoming far ice glorious counselors injector sum therapy cancer literature."     , n : "FACT CLAIM ACCUSATION"})
continuations232[8].push({ item : "232_Critical_8", s : "that the criminal who the stranger distracted baffled the officer seemed concerning.", a : "x-x-x occurs wild lot becoming sea low patience disordered stellar thy stopped cancer settlement."     , n : "FACT CLAIM ACCUSATION"})
continuations232[9].push({ item : "232_Critical_9", s : "that the customer who the vendor welcomed contacted the clerk was very believable.", a : "x-x-x slowly soil ago belonged bad joy marrow intimacy comforted bit solve sea zone archetypes."      , n : "CLAIM"})
continuations232[9].push({ item : "232_Critical_9", s : "that the customer who the vendor welcomed terrified the clerk was very believable.", a : "x-x-x broken gain nor wondered us fat arisen suppress southward mid cried sea root collieries."      , n : "CLAIM"})
continuations232[10].push({item : "232_Critical_10", s :  "that the president who the farmer admired appointed the commander was entirely bogus.", a : "x-x-x please wine ill gradually eat lot nobody custody mountains sum somewhere hot purposes heals."      , n : "CLAIM"})
continuations232[10].push({item : "232_Critical_10", s :  "that the president who the farmer admired impressed the commander was entirely bogus.", a : "x-x-x indeed thin sin including low eat verbal revival statutory sky dominated hot language tulip."      , n : "CLAIM"})
continuations232[11].push({item : "232_Critical_11", s :  "that the victim who the swimmer rescued sued the criminal appeared on TV.", a : "x-x-x itself soft die seldom sin old alimony beggars fuck sum diseases improve boy Era."     , n : "CLAIM FACT"})
continuations232[11].push({item : "232_Critical_11", s : "that the victim who the swimmer rescued surprised the criminal appeared on TV.", a : "x-x-x tissue cash thy afford sky thy backups stripes equations map instance improved sin Guy."     , n : "CLAIM FACT"})      
continuations232[12].push({item : "232_Critical_12", s : "that the guest who the cousin invited visited the uncle drove Jill crazy.", a : "x-x-x killed coal joy slept us net argues crucial classic map click teeth Stud brush."     , n : "CLAIM FACT"})
continuations232[12].push({item : "232_Critical_12", s : "that the guest who the cousin invited pleased the uncle drove Jill crazy.", a : "x-x-x myself sons eat funny sin ice behold crucial seventh dry click teeth Ores rigid."     , n : "CLAIM FACT"})
continuations232[13].push({item : "232_Critical_13", s : "that the psychiatrist who the nurse assisted diagnosed the patient became widely known.", a : "x-x-x inches gain lot competencies oil son quick blessing slaughter map replied divine manner forty."     , n : "CLAIM FACT"})
continuations232[13].push({item : "232_Critical_13", s : "that the psychiatrist who the nurse assisted horrified the patient became widely known.", a : "x-x-x accept slow lot disappearing ice sum quick symmetry nonverbal sky arrived divine create urban."     , n : "CLAIM FACT"})
continuations232[14].push({item : "232_Critical_14", s : "that the driver who the guide called phoned the tourist was absolutely true.", a : "x-x-x obtain foot eat tended its him below cities wreath fat oneself sin depression tube." , n : "CLAIM"})
continuations232[14].push({item : "232_Critical_14", s : "that the driver who the guide called amazed the tourist was absolutely true.", a : "x-x-x obtain warm nor afford its add since valley funded sun arrange sin hypothesis acid." , n : "CLAIM"})
continuations232[15].push({item : "232_Critical_15", s : "that the actor who the fans loved greeted the director appeared to be true.", a : "x-x-x quoted slow joy multi aid sin bury grace frankly net adequate addition sum off pair." , n : "CLAIM"})
continuations232[15].push({item : "232_Critical_15", s : "that the actor who the fans loved astonished the director appeared to be true.", a : "x-x-x hardly root ago eaten dry map sine child identities sky doctrine addition eat hot anti." , n : "CLAIM"})
continuations232[16].push({item : "232_Critical_16", s : "that the banker who the analyst defrauded trusted the customer proved to be made up.", a : "x-x-x slowly wind sky amazed eye age endured rainstorm soluble joy wherever spring eye few zone sat." , n : "CLAIM"})
continuations232[16].push({item : "232_Critical_16", s : "that the banker who the analyst defrauded excited the customer proved to be made up.", a : "x-x-x filled anti sin relies fat oil signify nightlife walking sin wondered spring sex eye thou sin." , n : "CLAIM"})
continuations232[17].push({item : "232_Critical_17", s : "that the judge who the attorney hated acquitted the defendant was a lie.", a : "x-x-x remove feet joy aware hot sum strictly strip populated bed everybody sum sum ice." , n : "CLAIM"})
continuations232[17].push({item : "232_Critical_17", s : "that the judge who the attorney hated vindicated the defendant was a lie.", a : "x-x-x anyone food lot until boy lot dynamics cream mobilizing big depending sum ago are." , n : "CLAIM"})
continuations232[18].push({item : "232_Critical_18", s : "that the captain who the crew trusted commanded the sailor was nice to hear.", a : "x-x-x obtain debt sky himself six add pray temples judgments fat echoed map wars sky goes.", n : "CLAIM FACT"})
continuations232[18].push({item : "232_Critical_18", s : "that the captain who the crew trusted motivated the sailor was nice to hear.", a : "x-x-x tissue root thy someone red end ears infants aesthetic sky graphs map runs age cell.", n : "CLAIM FACT"})
continuations232[19].push({item : "232_Critical_19", s : "that the manager who the boss authorized hired the intern seemed absurd.", a : "x-x-x walked soft die suppose thy mid bend continuing crazy sky skirts extent submit.", n : "CLAIM FACT"}) 
continuations232[19].push({item : "232_Critical_19", s : "that the manager who the boss authorized saddened the intern seemed absurd.", a : "x-x-x tissue easy eat another me fat span illustrate mainline map longed extent remedy.", n : "CLAIM FACT"}) 
continuations232[20].push({item : "232_Critical_20", s : "that the plaintiff who the jury interrogated interrupted the witness made it into the news.", a : "x-x-x hardly eyes eat depending its my mood subcontinent qualitative fat muscles iron eat wine lot dear.", n : "CLAIM FACT"})
continuations232[20].push({item : "232_Critical_20", s : "that the plaintiff who the jury interrogated startled the witness made it into the news.", a : "x-x-x walked foot eat everybody him mid tons governorship coherent joy assured iron buy ones sun your.", n : "CLAIM FACT"})
continuations232[21].push({item : "232_Critical_21", s : "that the drunkard who the thug hit outwitted the bartender sounded hilarious.", a : "x-x-x unable rise ill loudness mid bad raga par hairstyle joy musically squares vestments.", n : "CLAIM FACT"})
continuations232[21].push({item : "232_Critical_21", s : "that the drunkard who the thug hit stunned the bartender sounded hilarious.", a : "x-x-x unless slow die cucumber thy my raga par fissure joy transpose squares seafaring.", n : "CLAIM FACT"})
continuations232[22].push({exclude : true, item : "232_Critical_22", s : "that the pediatrician who the receptionist supported mistrusted the parent troubled people.", a : "x-x-x filled pure fat intensifying mid me overthrowing thousands recuperate mid hardly implicit impact.", n : "CLAIM FACT"})
continuations232[22].push({exclude : true, item : "232_Critical_22", s : "that the pediatrician who the receptionist supported disturbed the parent troubled people.", a : "x-x-x filled boat thy incarnations mid mid impressively important commodity joy myself implicit treaty.", n : "CLAIM FACT"})
continuations232[23].push({item : "232_Critical_23", s : "that the medic who the survivor thanked greeted the surgeon turned out to be untrue.", a : "x-x-x walked jobs ago vying sun end voltages swollen reddish thy manners annual sit sin per digits."  , n : "CLAIM"})
continuations232[23].push({item : "232_Critical_23", s : "that the medic who the survivor thanked surprised the surgeon turned out to be untrue.", a : "x-x-x expect mail sky evict age sum spacious reddish disorders end tonight annual ago map sat walled."  , n : "CLAIM"})
continuations232[24].push({item : "232_Critical_24", s : "that the lifesaver who the soldier taught rescued the swimmer took the townspeople by surprise.", a : "x-x-x unable rise eat befriends thy son examine appeal invoice map amplify soil ago indivisible mid theology.", n : "CLAIM FACT"})
continuations232[24].push({item : "232_Critical_24", s : "that the lifesaver who the soldier taught encouraged the swimmer took the townspeople by surprise.", a : "x-x-x issued hear ago pressings thy us walking origin occasional bed groomed soil ago naturalized sin merchant.", n : "CLAIM FACT"})
continuations232[25].push({item : "232_Critical_25", s : "that the fisherman who the gardener helped admired the politician was interesting.", a : "x-x-x unable warm lot laterally sin why platelet remove labeled sit ministries sky possibility.", n : "CLAIM FACT"})
continuations232[25].push({item : "232_Critical_25", s : "that the fisherman who the gardener helped delighted the politician was interesting.", a : "x-x-x happen fine sky treachery why red alarming months whispered map wavelength sky possibility.", n : "CLAIM FACT"})
continuations232[26].push({item : "232_Critical_26", s : "that the janitor who the organizer criticized ignored the audience was funny.", a : "x-x-x expect feet sky recluse us big vertebral insistence optical buy entitled thy ridge.", n : "CLAIM FACT"})
continuations232[26].push({item : "232_Critical_26", s : "that the janitor who the organizer criticized amused the audience was funny.", a : "x-x-x issued warm nor encodes why fat smoothing responsive mortar mid composed thy spare.", n : "CLAIM FACT"})
continuations232[27].push({item : "232_Critical_27", s : "that the investor who the scientist detested deceived the entrepreneur drove everyone crazy.", a : "x-x-x toward neck ago hesitate sit add imperfect eclectic grandeur fat helplessness beach remained tries.", n : "CLAIM FACT"})
continuations232[27].push({item : "232_Critical_27", s : "that the investor who the scientist detested disappointed the entrepreneur drove everyone crazy.", a : "x-x-x remove mere ill wandered fat sex differing amicable anthropology bed unrestricted beach doctrine renumbering.", n : "CLAIM FACT"})
continuations232[28].push({item : "232_Critical_28", s : "that the firefighter who the neighbor insulted rescued the houseowner went unnoticed.", a : "x-x-x taught edge lot blackmailed joy net resemble puncture summers bit captaining wise iteration.", n : "CLAIM FACT"})
continuations232[28].push({item : "232_Critical_28", s : "that the firefighter who the neighbor insulted discouraged the houseowner went unnoticed.", a : "x-x-x occurs duty nor blackmailed thy how avoiding auditors reminiscent sky premiering wise forecasts.", n : "CLAIM FACT"})
continuations232[29].push({exclude : true, item : "232_Critical_29", s : "that the vendor who the storeowner recruited welcomed the client excited the boss.", a : "x-x-x become else ill summed sea sun mothballed awakening pleading sky hardly justify mid coil.", n : "CLAIM FACT"})
continuations232[29].push({exclude : true, item : "232_Critical_29", s : "that the vendor who the storeowner recruited satisfied the client excited the boss.", a : "x-x-x occurs wish nor pillow sky ten weeknights utterance gentleman sky filled statute sit dies.", n : "CLAIM FACT"})
continuations232[30].push({item : "232_Critical_30", s : "that the plumber who the apprentice consulted assisted the woman was true.", a : "x-x-x issued dear fat scoured its ten appointing gratitude reliance mid exist eat spot."  , n : "CLAIM"})
continuations232[30].push({item : "232_Critical_30", s : "that the plumber who the apprentice consulted puzzled the woman was true.", a : "x-x-x issued sand die tracery mid mid humanities certainty linkage joy apply sun arts."  , n : "CLAIM"})
continuations232[31].push({exclude : true, item : "232_Critical_31", s : "that the sponsor who the musician entertained cheered the onlookers pleased everyone.", a : "x-x-x quoted fine fat liquids sky dry auditory particulars seismic end porcupine statute elements.", n : "CLAIM FACT"})
continuations232[31].push({exclude : true, item : "232_Critical_31", s : "that the sponsor who the musician entertained captivated the onlookers pleased everyone.", a : "x-x-x anyone lose joy descend me nor harmless formulation victimized non urination circles negative.", n : "CLAIM FACT"})



for(i=0; i<32; i++) {
	for(j=0; j<2; j++) {
		s = continuations232[i][j].s.split(" ");
		continuations232[i][j].r = "REGION_0_0 REGION_0_1 REGION_0_2 REGION_1_0 REGION_1_1 REGION_1_2 REGION_1_3 REGION_2_0 REGION_2_1 REGION_2_2 REGION_3_0 REGION_3_1 REGION_3_2 REGION_3_3 REGION_4_0"
	}
}


//continuations = continuations238.concat(continuations232);
////// Specifically select some continuations
continuations232_ = [];
for(i=0; i < continuations232.length; i++) {
	if(continuations232[i][0].exclude != true) {
		
		continuations232_.push(continuations232[i]);
	}
}
continuations232 = continuations232_;

continuations238_ = [];
for(i=0; i < continuations238.length; i++) {
	console.log(i);
	console.log(continuations238[i]);
	console.log(continuations238[i][0].exclude);
	if(continuations238[i][0].exclude != true) {
		continuations238_.push(continuations238[i]);
	}
}
continuations238 = continuations238_;


continuations = continuations238.concat(continuations232);

console.log(continuations);



for(i = 0; i<continuations.length; i++) {
	console.log(i, continuations[i]);
	for(j =0; j<2; j++) {
                continuations[i][j].condition = "continuations"
		continuations[i][j].n = continuations[i][j].n.split(" ")
	}
}



// Now specifically select the old continuations
//continuations = continuations.slice(0, 20);

continuationsChosen = _.shuffle(continuations);
console.log(continuationsChosen);





console.log(topNouns);

FAILED = true;
matching_attempts = 0;
while(FAILED) {
topNouns = _.shuffle(topNouns);

matching_attempts = matching_attempts+1;
	if(matching_attempts > 10) {
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
		if(conditionAssignment[i] == 0) { 
    		    item["condition"] = "critical_incompatible"
		} else if(conditionAssignment[i] == 1) {
    		    item["condition"] = "critical_NoSC" 
		} else if (conditionAssignment[i] == -1) {
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
			if((!(continuations[i][0].item.includes("232"))) && j < regions.length && regions[j] == "REGION_3_0") {
				if(Math.random() > 0.5) {
					item.distractor_condition = "dist1";
					distractors.push(distractors1[j]);
				} else {
					item.distractor_condition = "dist2";
					distractors.push(distractors2[j]);
				}
			} else {
				distractors.push(distractors1[j]);
			}
		}
		item.a = distractors.join(" ");
	} else {
		item = continuations[i][1];
		if(conditionAssignment[i] == 2) {
  		    item["condition"] = "critical_compatible"
		} else if(conditionAssignment[i] == 3) {
			item["condition"] = "critical_SCRC_compatible"
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
                 if(!(continuations[i][0].item.includes("232"))) {
                  if(j < regions.length && regions[j] == "REGION_3_0") { // distractor on the critical verb
                    if(Math.random() > 0.5) { // This random choice is taken over from the code of Study S5, but here distractors actually are matched across the compatibility manipulation, so the random choice plays no real role
                         item.distractor_condition = "dist1";
                         distractors.push(distractors1[j]);
                    } else {
                         item.distractor_condition = "dist2";
                         distractors.push(distractors2[j]);
                    }
                  } else if(j >= words1.length || words1[j] != words2[j]) { // distractor specific to this version
                    distractors.push(distractors2[j]);
                  } else if(words2[j] == words1[j]) { // else, match across conditions
                    distractors.push(distractors1[j]);
                  }
                } else {
                  if(j < regions.length && (words1[j] != words2[j]) || (j > 0 && words1[j-1] != words2[j-1])) { // distractor specific to this version
                    distractors.push(distractors2[j]);
                  } else { // else, match across conditions
                    distractors.push(distractors1[j]);
                  }
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
	result = indices_x.concat(indices_y);
	attempts_order = 0;
	console.log("SHUFFLING");
	result = _.shuffle(result);
	for(i=0; i+1<result.length; i++) {
		if(result[i][0] == "y" && result[i+1][0] == "y") {
			candidate_positions = [];
	                for(j=0; j+2<result.length; j++) {
                           if(result[j][0] == "x" && result[j+1][0] == "x" && result[j+2][0] == "x") {
				   candidate_positions.push(j+1);
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

fullStimuli = _.shuffle(practice).concat(fillersAndCritical);

item_ids = [];

console.log( fullStimuli);
return fullStimuli;
     
}

