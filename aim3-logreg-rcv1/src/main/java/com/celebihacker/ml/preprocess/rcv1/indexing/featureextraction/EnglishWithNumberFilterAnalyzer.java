package com.celebihacker.ml.preprocess.rcv1.indexing.featureextraction;

import java.io.IOException;
import java.io.Reader;
import java.nio.CharBuffer;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenFilter;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.core.LowerCaseFilter;
import org.apache.lucene.analysis.core.StopFilter;
import org.apache.lucene.analysis.en.EnglishPossessiveFilter;
import org.apache.lucene.analysis.en.PorterStemFilter;
import org.apache.lucene.analysis.standard.StandardFilter;
import org.apache.lucene.analysis.standard.StandardTokenizer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.util.CharArraySet;
import org.apache.lucene.util.Version;

import com.google.common.collect.ImmutableSet;

/**
 * RCV1Analyzer behaves like EnglishAnalyzer and adds filtering of numbers.
 * 
 * Stop words are taken from online appendix 11 (http://jmlr.org/papers/volume5/lewis04a/lyrl2004_rcv1v2_README.htm) of
 * [1].
 * 
 * [1] David D. Lewis, Yiming Yang, Tony G. Rose, and Fan Li. 2004. RCV1: A New Benchmark Collection for Text
 * Categorization Research. J. Mach. Learn. Res. 5 (December 2004), 361-397.
 */
public class EnglishWithNumberFilterAnalyzer extends Analyzer {

    public enum NumberFilterMethod {
        KEEP(null), // Keep numbers
        REMOVE(Pattern.compile("^[0-9,.]+$")), // Remove numbers
        ROUND(Pattern.compile("^([0-9]+)([,.0-9a-z]*)$")); // Remove part after ',' or '.'

        public final Pattern filterPattern;

        NumberFilterMethod(Pattern pattern) {
            this.filterPattern = pattern;
        }
    }

    private static final Set<String> STOP_WORDS =
            new ImmutableSet.Builder<String>()
                    .add("a").add("a's").add("able").add("about").add("above")
                    .add("according").add("accordingly").add("across")
                    .add("actually").add("after").add("afterwards")
                    .add("again").add("against").add("ain't").add("all")
                    .add("allow").add("allows").add("almost").add("alone")
                    .add("along").add("already").add("also").add("although")
                    .add("always").add("am").add("among").add("amongst")
                    .add("an").add("and").add("another").add("any")
                    .add("anybody").add("anyhow").add("anyone").add("anything")
                    .add("anyway").add("anyways").add("anywhere").add("apart")
                    .add("appear").add("appreciate").add("appropriate")
                    .add("are").add("aren't").add("around").add("as")
                    .add("aside").add("ask").add("asking").add("associated")
                    .add("at").add("available").add("away").add("awfully")
                    .add("b").add("be").add("became").add("because")
                    .add("become").add("becomes").add("becoming").add("been")
                    .add("before").add("beforehand").add("behind").add("being")
                    .add("believe").add("below").add("beside").add("besides")
                    .add("best").add("better").add("between").add("beyond")
                    .add("both").add("brief").add("but").add("by").add("c")
                    .add("c'mon").add("c's").add("came").add("can")
                    .add("can't").add("cannot").add("cant").add("cause")
                    .add("causes").add("certain").add("certainly")
                    .add("changes").add("clearly").add("co").add("com")
                    .add("come").add("comes").add("concerning")
                    .add("consequently").add("consider").add("considering")
                    .add("contain").add("containing").add("contains")
                    .add("corresponding").add("could").add("couldn't")
                    .add("course").add("currently").add("d").add("definitely")
                    .add("described").add("despite").add("did").add("didn't")
                    .add("different").add("do").add("does").add("doesn't")
                    .add("doing").add("don't").add("done").add("down")
                    .add("downwards").add("during").add("e").add("each")
                    .add("edu").add("eg").add("eight").add("either")
                    .add("else").add("elsewhere").add("enough").add("entirely")
                    .add("especially").add("et").add("etc").add("even")
                    .add("ever").add("every").add("everybody").add("everyone")
                    .add("everything").add("everywhere").add("ex")
                    .add("exactly").add("example").add("except").add("f")
                    .add("far").add("few").add("fifth").add("first")
                    .add("five").add("followed").add("following")
                    .add("follows").add("for").add("former").add("formerly")
                    .add("forth").add("four").add("from").add("further")
                    .add("furthermore").add("g").add("get").add("gets")
                    .add("getting").add("given").add("gives").add("go")
                    .add("goes").add("going").add("gone").add("got")
                    .add("gotten").add("greetings").add("h").add("had")
                    .add("hadn't").add("happens").add("hardly").add("has")
                    .add("hasn't").add("have").add("haven't").add("having")
                    .add("he").add("he's").add("hello").add("help")
                    .add("hence").add("her").add("here").add("here's")
                    .add("hereafter").add("hereby").add("herein")
                    .add("hereupon").add("hers").add("herself").add("hi")
                    .add("him").add("himself").add("his").add("hither")
                    .add("hopefully").add("how").add("howbeit").add("however")
                    .add("i").add("i'd").add("i'll").add("i'm").add("i've")
                    .add("ie").add("if").add("ignored").add("immediate")
                    .add("in").add("inasmuch").add("inc").add("indeed")
                    .add("indicate").add("indicated").add("indicates")
                    .add("inner").add("insofar").add("instead").add("into")
                    .add("inward").add("is").add("isn't").add("it").add("it'd")
                    .add("it'll").add("it's").add("its").add("itself").add("j")
                    .add("just").add("k").add("keep").add("keeps").add("kept")
                    .add("know").add("knows").add("known").add("l").add("last")
                    .add("lately").add("later").add("latter").add("latterly")
                    .add("least").add("less").add("lest").add("let")
                    .add("let's").add("like").add("liked").add("likely")
                    .add("little").add("look").add("looking").add("looks")
                    .add("ltd").add("m").add("mainly").add("many").add("may")
                    .add("maybe").add("me").add("mean").add("meanwhile")
                    .add("merely").add("might").add("more").add("moreover")
                    .add("most").add("mostly").add("much").add("must")
                    .add("my").add("myself").add("n").add("name").add("namely")
                    .add("nd").add("near").add("nearly").add("necessary")
                    .add("need").add("needs").add("neither").add("never")
                    .add("nevertheless").add("new").add("next").add("nine")
                    .add("no").add("nobody").add("non").add("none")
                    .add("noone").add("nor").add("normally").add("not")
                    .add("nothing").add("novel").add("now").add("nowhere")
                    .add("o").add("obviously").add("of").add("off")
                    .add("often").add("oh").add("ok").add("okay").add("old")
                    .add("on").add("once").add("one").add("ones").add("only")
                    .add("onto").add("or").add("other").add("others")
                    .add("otherwise").add("ought").add("our").add("ours")
                    .add("ourselves").add("out").add("outside").add("over")
                    .add("overall").add("own").add("p").add("particular")
                    .add("particularly").add("per").add("perhaps")
                    .add("placed").add("please").add("plus").add("possible")
                    .add("presumably").add("probably").add("provides").add("q")
                    .add("que").add("quite").add("qv").add("r").add("rather")
                    .add("rd").add("re").add("really").add("reasonably")
                    .add("regarding").add("regardless").add("regards")
                    .add("relatively").add("respectively").add("right")
                    .add("s").add("said").add("same").add("saw").add("say")
                    .add("saying").add("says").add("second").add("secondly")
                    .add("see").add("seeing").add("seem").add("seemed")
                    .add("seeming").add("seems").add("seen").add("self")
                    .add("selves").add("sensible").add("sent").add("serious")
                    .add("seriously").add("seven").add("several").add("shall")
                    .add("she").add("should").add("shouldn't").add("since")
                    .add("six").add("so").add("some").add("somebody")
                    .add("somehow").add("someone").add("something")
                    .add("sometime").add("sometimes").add("somewhat")
                    .add("somewhere").add("soon").add("sorry").add("specified")
                    .add("specify").add("specifying").add("still").add("sub")
                    .add("such").add("sup").add("sure").add("t").add("t's")
                    .add("take").add("taken").add("tell").add("tends")
                    .add("th").add("than").add("thank").add("thanks")
                    .add("thanx").add("that").add("that's").add("thats")
                    .add("the").add("their").add("theirs").add("them")
                    .add("themselves").add("then").add("thence").add("there")
                    .add("there's").add("thereafter").add("thereby")
                    .add("therefore").add("therein").add("theres")
                    .add("thereupon").add("these").add("they").add("they'd")
                    .add("they'll").add("they're").add("they've").add("think")
                    .add("third").add("this").add("thorough").add("thoroughly")
                    .add("those").add("though").add("three").add("through")
                    .add("throughout").add("thru").add("thus").add("to")
                    .add("together").add("too").add("took").add("toward")
                    .add("towards").add("tried").add("tries").add("truly")
                    .add("try").add("trying").add("twice").add("two").add("u")
                    .add("un").add("under").add("unfortunately").add("unless")
                    .add("unlikely").add("until").add("unto").add("up")
                    .add("upon").add("us").add("use").add("used").add("useful")
                    .add("uses").add("using").add("usually").add("uucp")
                    .add("v").add("value").add("various").add("very")
                    .add("via").add("viz").add("vs").add("w").add("want")
                    .add("wants").add("was").add("wasn't").add("way").add("we")
                    .add("we'd").add("we'll").add("we're").add("we've")
                    .add("welcome").add("well").add("went").add("were")
                    .add("weren't").add("what").add("what's").add("whatever")
                    .add("when").add("whence").add("whenever").add("where")
                    .add("where's").add("whereafter").add("whereas")
                    .add("whereby").add("wherein").add("whereupon")
                    .add("wherever").add("whether").add("which").add("while")
                    .add("whither").add("who").add("who's").add("whoever")
                    .add("whole").add("whom").add("whose").add("why")
                    .add("will").add("willing").add("wish").add("with")
                    .add("within").add("without").add("won't").add("wonder")
                    .add("would").add("would").add("wouldn't").add("x")
                    .add("y").add("yes").add("yet").add("you").add("you'd")
                    .add("you'll").add("you're").add("you've").add("your")
                    .add("yours").add("yourself").add("yourselves").add("z")
                    .add("zero").build();

    private final Version version;

    private final CharArraySet stopwords;

    final NumberFilterMethod numberFilterMethod;

    public EnglishWithNumberFilterAnalyzer() {
        this(NumberFilterMethod.KEEP);
    }

    public EnglishWithNumberFilterAnalyzer(NumberFilterMethod numberFilterMethod) {

        this.version = Version.LUCENE_43;
        this.stopwords = new CharArraySet(this.version, STOP_WORDS, true);
        this.numberFilterMethod = numberFilterMethod;

    }

    @Override
    protected TokenStreamComponents createComponents(String fieldName, Reader reader) {
        final Tokenizer source = new StandardTokenizer(this.version, reader);

        TokenStream result = new StandardFilter(this.version, source);
        result = new EnglishPossessiveFilter(this.version, result);
        result = new LowerCaseFilter(this.version, result);
        result = new StopFilter(this.version, result, this.stopwords);
        result = new PorterStemFilter(result);
        result = new NumberFilter(result, this.numberFilterMethod);

        return new TokenStreamComponents(source, result);
    }

    public class NumberFilter extends TokenFilter {

        private final CharTermAttribute term = addAttribute(CharTermAttribute.class);

        private final NumberFilterMethod filterMethod;

        private final Pattern filterPattern;

        protected NumberFilter(TokenStream input, NumberFilterMethod filterMethod) {
            super(input);

            this.filterMethod = filterMethod;
            this.filterPattern = filterMethod.filterPattern;
        }

        @Override
        public boolean incrementToken() throws IOException {
            if (this.input.incrementToken()) {
                if (this.filterMethod == NumberFilterMethod.KEEP)
                    return true;

                CharBuffer buffer = CharBuffer.wrap(this.term.buffer(), 0, this.term.length());
                Matcher filterMatcher = this.filterPattern.matcher(buffer);

                if (this.filterMethod == NumberFilterMethod.REMOVE)
                    return !filterMatcher.matches();

                if (this.filterMethod == NumberFilterMethod.ROUND && filterMatcher.matches())
                    this.term.setLength(filterMatcher.end(1));

                return true;
            }

            return false;
        }
    }
}