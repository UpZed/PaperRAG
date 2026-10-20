package com.yizhaoqi.smartpai.entity;

import lombok.Getter;
import lombok.Setter;

// 文件分块内容实体类
@Setter
@Getter
public class TextChunk {

    // Getters/Setters
    private int chunkId;       // 分块序号
    private String content;    // 分块内容
    private String docTitle;   // 文档标题
    private String sectionTitle; // 章节标题
    private Integer sectionLevel; // 章节层级

    // 构造方法
    public TextChunk(int chunkId, String content) {
        this(chunkId, content, null, null, null);
    }

    // 完整构造方法
    public TextChunk(int chunkId, String content, String docTitle, String sectionTitle, Integer sectionLevel) {
        this.chunkId = chunkId;
        this.content = content;
        this.docTitle = docTitle;
        this.sectionTitle = sectionTitle;
        this.sectionLevel = sectionLevel;
    }
}
