package de.tuberlin.dima.ml.preprocess.rcv1.indexing.parsing;

import java.util.Collection;
import java.util.HashSet;
import java.util.Set;

import com.google.common.collect.Multimap;

import de.tuberlin.dima.ml.preprocess.rcv1.indexing.types.NewsItem;
import de.tuberlin.dima.ml.preprocess.rcv1.indexing.types.RCV1;

/**
 * Patch RCV1-v1 articles to RCV1-v2.
 * 
 * [1] David D. Lewis, Yiming Yang, Tony G. Rose, and Fan Li. 2004. RCV1: A New Benchmark Collection for Text
 * Categorization Research. J. Mach. Learn. Res. 5 (December 2004), 361-397.
 * 
 * See http://jmlr.org/papers/volume5/lewis04a/lyrl2004_rcv1v2_README.htm for the online appendices, which describe
 * RCV1-v2.
 */
public final class NewsItemV2Patcher {

    /**
     * Patch RCV1-v1 article to RCV1-v2 according to Section 4 (p. 19) of [1].
     * 
     * 2377 articles will be removed (i.e. return false), because of Minimum Code Policy violations. For some of the
     * kept articles (i.e. return true), missing ancestors of topic codes are inserted and (three) incorrect region
     * codes are updated.
     * 
     * @param codes
     *            Multimap of 'codes' tag of XML file. Key is 'class' attribute and values are respective codes, e.g.
     *            "bip:topics:1.0" -> ["E11", ..., "MCAT"]
     * @return true, if article codes agree to RCV1-v2 after patching, false otherwise
     */
    public static boolean patch(Multimap<String, String> codes) {

        // 1. Check for minimum code policy violation (-2377 articles)
        if (!codes.containsKey(RCV1.TOPIC_CODE) || !codes.containsKey(RCV1.REGION_CODE))
            return false;

        // 2. Add all missing ancestors of each topic code
        Collection<String> topics = codes.get(RCV1.TOPIC_CODE);
        Set<String> toAdd = new HashSet<String>();

        for (String t : topics) {
            for (String ancestor : RCV1.TOPIC_ANCESTORS.get(t)) {
                if (!topics.contains(ancestor)) {
                    toAdd.add(ancestor);
                }
            }
        }

        codes.putAll(RCV1.TOPIC_CODE, toAdd);

        // 3. Fix incorrect region codes (see section 3.2.3 (p. 14) of [1])
        // Note: the article with region code 'CZ' will have 'PANA' two times after the fix. This
        // seems odd, but is in agreement with Appendix 10 of [1].
        if (codes.containsEntry(RCV1.REGION_CODE, "CZ")) {

            codes.remove(RCV1.REGION_CODE, "CZ");
            codes.put(RCV1.REGION_CODE, "PANA");

        } else if (codes.containsEntry(RCV1.REGION_CODE, "CZECH")) {

            codes.remove(RCV1.REGION_CODE, "CZECH");
            codes.put(RCV1.REGION_CODE, "CZREP");

        } else if (codes.containsEntry(RCV1.REGION_CODE, "GDR")) {
            codes.remove(RCV1.REGION_CODE, "GDR");
            codes.put(RCV1.REGION_CODE, "GFR");
        }

        return true;

    }

    /**
     * Patch RCV1-v1 NewsItem object to RCV1-v2 according to section 4 (p. 19) of [1].
     * 
     * @param newsItem
     *            NewsItem to patch
     * @return true, if NewsItem agrees to RCV1-v2 after patching, false otherwise
     */
    public static boolean patch(NewsItem newsItem) {

        return patch(newsItem.codes());

    }
}