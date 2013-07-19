package de.tuberlin.dima.ml.writables;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.WritableComparable;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.Vector.Element;

import com.google.common.primitives.Ints;

/**
 * For categorical labels, that were exploded to binary labels
 * Each binary label is assumed to be either 0 or 1
 * 
 * A mapping from label-id to label-name can be described using DatasetInfo
 */
public class IDAndLabels implements WritableComparable<IDAndLabels> {

    private int id;
    private Vector labels;

    public void set(int id, Vector labels) {
        this.id = id;
        this.labels = labels;
    }

    public int getId() {
        return this.id;
    }

    public Vector getLabels() {
        return this.labels;
    }

    public int compareTo(IDAndLabels o) {
        return Ints.compare(this.id, o.id);
    }

    public void write(DataOutput out)
            throws IOException {
        out.writeInt(id);
        out.writeInt(labels.size());

        for (Element label : labels.all()) {
            out.writeDouble(label.get());
        }
    }

    public void readFields(DataInput in)
            throws IOException {
        this.id = in.readInt();
        int numCodes = in.readInt();
        
        this.labels = new DenseVector(numCodes);
        for (int index = 0; index < numCodes; index++) {
            this.labels.set(index, in.readDouble());
        }
    }
}