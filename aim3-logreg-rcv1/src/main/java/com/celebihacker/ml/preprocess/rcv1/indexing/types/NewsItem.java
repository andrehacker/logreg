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

package com.celebihacker.ml.preprocess.rcv1.indexing.types;

import com.google.common.collect.Multimap;

import java.util.Date;
import java.util.Map;

public class NewsItem {

    private final int itemID;

    private final Date date;

    private final String title;
    private final String headline;
    private final String text;
    private final String dateline;

    private final Multimap<String, String> codes;
    private final Map<String, String> dcs;

    public NewsItem(int itemID, Date date, String title, String headline,
            String text, String dateline, Multimap<String, String> codes,
            Map<String, String> dcs) {

        this.itemID = itemID;
        this.date = date;
        this.title = title;
        this.headline = headline;
        this.text = text;
        this.dateline = dateline;
        this.codes = codes;
        this.dcs = dcs;

    }

    public int itemID() {
        return this.itemID;
    }

    public Date date() {
        return this.date;
    }

    public String title() {
        return this.title;
    }

    public String headline() {
        return this.headline;
    }

    public String text() {
        return this.text;
    }

    public String dateline() {
        return this.dateline;
    }

    public Multimap<String, String> codes() {
        return this.codes;
    }

    public Map<String, String> dcs() {
        return this.dcs;
    }
}