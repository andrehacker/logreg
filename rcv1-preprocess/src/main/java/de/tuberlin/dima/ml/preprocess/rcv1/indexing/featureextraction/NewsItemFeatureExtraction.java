package de.tuberlin.dima.ml.preprocess.rcv1.indexing.featureextraction;

import java.text.SimpleDateFormat;
import java.util.Collection;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.IntField;
import org.apache.lucene.document.StringField;

import de.tuberlin.dima.ml.preprocess.rcv1.indexing.types.NewsItem;
import de.tuberlin.dima.ml.preprocess.rcv1.indexing.types.RCV1;
import de.tuberlin.dima.ml.preprocess.rcv1.indexing.types.TextFieldWithTermVectors;

/**
 * Extract features as Lucene Document objects from parsed NewsItem objects.
 * 
 * Use 'title' and 'text' fields as features and top-level topic codes as labels.
 */
public class NewsItemFeatureExtraction {

    public static final String ITEM_ID = "itemId";
    public static final String DATE = "date";
    public static final String TEXT = "text";
    public static final String TOPICS = "topics";

    private final SimpleDateFormat DATE_FORMAT = new SimpleDateFormat("yyyyMMdd");

    private final Document doc = new Document();
    private final IntField id = new IntField(ITEM_ID, 0, Field.Store.YES);
    private final StringField date = new StringField(DATE, "", Field.Store.NO);
    private final TextFieldWithTermVectors text = new TextFieldWithTermVectors(TEXT, "");
    private final TextFieldWithTermVectors topics = new TextFieldWithTermVectors(TOPICS, "");

    public NewsItemFeatureExtraction() {
        this.doc.add(this.id);
        this.doc.add(this.date);
        this.doc.add(this.text);
        this.doc.add(this.topics);
    }

    public Document extract(NewsItem item) {
        // Only keep top-level topics 
        Collection<String> topics = item.codes().get(RCV1.TOPIC_CODE);
        topics.retainAll(RCV1.TOP_LEVEL_TOPICS.keySet());

        // Update document fields
        this.id.setIntValue(item.itemID());
        this.date.setStringValue(this.DATE_FORMAT.format(item.date()).toString());
        this.text.setStringValue(item.title() + " " + item.text());
        this.topics.setStringValue(topics.toString());

        return this.doc;
    }
}