package com.celebihacker.ml.preprocess.rcv1.indexing.types;

import java.util.Map;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.Multimap;

/**
 * RCV1-specific information.
 * 
 * [1] David D. Lewis, Yiming Yang, Tony G. Rose, and Fan Li. 2004. RCV1: A New Benchmark Collection for Text
 * Categorization Research. J. Mach. Learn. Res. 5 (December 2004), 361-397.
 * 
 * See http://jmlr.org/papers/volume5/lewis04a/lyrl2004_rcv1v2_README.htm for the online appendices of [1].
 */
public class RCV1 {

    /**
     * Complete mapping from sub-topics to ancestors (except for top level topics with parent root).
     * 
     * This mapping is based on online appendix 2 of [1].
     */
    public static final Multimap<String, String> TOPIC_ANCESTORS =
            new ImmutableMultimap.Builder<String, String>()
                    .put("C11", "CCAT").put("C12", "CCAT").put("C13", "CCAT").put("C14", "CCAT")
                    .put("C15", "CCAT").putAll("C151", "C15", "CCAT")
                    .putAll("C1511", "C151", "C15", "CCAT").putAll("C152", "C15", "CCAT")
                    .put("C16", "CCAT").put("C17", "CCAT").putAll("C171", "C17", "CCAT")
                    .putAll("C172", "C17", "CCAT").putAll("C173", "C17", "CCAT")
                    .putAll("C174", "C17", "CCAT").put("C18", "CCAT").putAll("C181", "C18", "CCAT")
                    .putAll("C182", "C18", "CCAT").putAll("C183", "C18", "CCAT").put("C21", "CCAT")
                    .put("C22", "CCAT").put("C23", "CCAT").put("C24", "CCAT").put("C31", "CCAT")
                    .putAll("C311", "C31", "CCAT").putAll("C312", "C31", "CCAT")
                    .putAll("C313", "C31", "CCAT").put("C32", "CCAT").put("C33", "CCAT")
                    .putAll("C331", "C33", "CCAT").put("C34", "CCAT").put("C41", "CCAT")
                    .putAll("C411", "C41", "CCAT").put("C42", "CCAT").put("E11", "ECAT").put("E12", "ECAT")
                    .putAll("E121", "E12", "ECAT").put("E13", "ECAT").putAll("E131", "E13", "ECAT")
                    .putAll("E132", "E13", "ECAT").put("E14", "ECAT").putAll("E141", "E14", "ECAT")
                    .putAll("E142", "E14", "ECAT").putAll("E143", "E14", "ECAT").put("E21", "ECAT")
                    .putAll("E211", "E21", "ECAT").putAll("E212", "E21", "ECAT").put("E31", "ECAT")
                    .putAll("E311", "E31", "ECAT").putAll("E312", "E31", "ECAT")
                    .putAll("E313", "E31", "ECAT").put("E41", "ECAT").putAll("E411", "E41", "ECAT")
                    .put("E51", "ECAT").putAll("E511", "E51", "ECAT").putAll("E512", "E51", "ECAT")
                    .putAll("E513", "E51", "ECAT").put("E61", "ECAT").put("E71", "ECAT").put("G15", "GCAT")
                    .putAll("G151", "G15", "GCAT").putAll("G152", "G15", "GCAT")
                    .putAll("G153", "G15", "GCAT").putAll("G154", "G15", "GCAT")
                    .putAll("G155", "G15", "GCAT").putAll("G156", "G15", "GCAT")
                    .putAll("G157", "G15", "GCAT").putAll("G158", "G15", "GCAT")
                    .putAll("G159", "G15", "GCAT").put("GCRIM", "GCAT").put("GDEF", "GCAT")
                    .put("GDIP", "GCAT").put("GDIS", "GCAT").put("GENT", "GCAT").put("GENV", "GCAT")
                    .put("GFAS", "GCAT").put("GHEA", "GCAT").put("GJOB", "GCAT").put("GMIL", "GCAT")
                    .put("GOBIT", "GCAT").put("GODD", "GCAT").put("GPOL", "GCAT").put("GPRO", "GCAT")
                    .put("GREL", "GCAT").put("GSCI", "GCAT").put("GSPO", "GCAT").put("GTOUR", "GCAT")
                    .put("GVIO", "GCAT").put("GVOTE", "GCAT").put("GWEA", "GCAT").put("GWELF", "GCAT")
                    .put("M11", "MCAT").put("M12", "MCAT").put("M13", "MCAT").putAll("M131", "M13", "MCAT")
                    .putAll("M132", "M13", "MCAT").put("M14", "MCAT").putAll("M141", "M14", "MCAT")
                    .putAll("M142", "M14", "MCAT").putAll("M143", "M14", "MCAT")
                    .build();

    /**
     * Set of top level topics (with parent root). Every documents in RCV1-v2 has at least one of these topics assigned.
     */
    public static final Map<String, Integer> TOP_LEVEL_TOPICS =
            new ImmutableMap.Builder<String, Integer>()
                    .put("CCAT", 0)
                    .put("ECAT", 1)
                    .put("GCAT", 2)
                    .put("MCAT", 3)
                    .build();

    /** Class of topic codes in XML documents */
    public static final String TOPIC_CODE = "bip:topics:1.0";

    /** Class of region codes in XML documents */
    public static final String REGION_CODE = "bip:countries:1.0";

}