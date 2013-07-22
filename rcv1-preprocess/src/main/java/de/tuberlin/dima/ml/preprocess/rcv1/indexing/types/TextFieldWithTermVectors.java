/*
 * Copyright (C) 2013 Database Systems and Information Management Group, TU
 * Berlin
 * 
 * cuttlefish is free software; you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation; either version 2 of the License, or (at your option) any later
 * version.
 * 
 * cuttlefish is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along with
 * cuttlefish; if not, write to the Free Software Foundation, Inc., 59 Temple
 * Place, Suite 330, Boston, MA 02111-1307 USA
 */

package de.tuberlin.dima.ml.preprocess.rcv1.indexing.types;

import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.index.FieldInfo;

public class TextFieldWithTermVectors extends Field {

    public static final FieldType TYPE = new FieldType();

    static {
        TYPE.setIndexed(true);
        TYPE.setOmitNorms(true);
        TYPE.setIndexOptions(FieldInfo.IndexOptions.DOCS_AND_FREQS);
        TYPE.setTokenized(true);
        TYPE.setStoreTermVectors(true);
        TYPE.freeze();
    }

    public TextFieldWithTermVectors(String name, String value) {
        super(name, value, TYPE);
    }
}