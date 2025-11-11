# -*- coding = utf-8 -*-
# @Time : 2025/11/6 8:23 下午
# @Author Sun Jingchen
# @File nlp_predict.py
# @Software PyCharm
# -*- coding: utf-8 -*-
# version:2024.8.8
# 2024.8月以后竞赛必须使用该版本
# 项目名称:客户标签画像分析

###请在这里import想调用的库
import numpy as np

import paddle
from paddlenlp.transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
)


def read_texts_dataset(texts, is_test):
    for text in texts:
        parts = list(map(lambda s: s.strip(), text.split("\t")))
        sentence = "\t".join(parts[:-1])
        labels = parts[-1]
        if is_test:
            yield {"sentence": sentence, "labels": labels}
        else:
            yield {"sentence": sentence}


class Predictor:
    """
    InitModel函数  模型初始化参数,注意不能自行增加删除函数入参
    ret            是否正常: 正常True,异常False
    err_message    错误信息: 默认normal
    return ret,err_message
    """

    def InitModel(self):
        ret = True
        err_message = "normal"
        """
        模型初始化,由用户自行编写
        加载出错时给ret和err_message赋值相应的错误
        *注意模型应为相对路径
        """
        ### 请在try内编写模型初始化,便于捕获错误
        try:
            ##########模型初始化开始##########
            ### 加载模型
            ### self.model=load_model(model_path)
            self.device = "gpu"
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "checkpoint"
            ).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained("checkpoint")
            self.max_seq_length = 512
            self.batch_size = 1

            config = AutoConfig.from_pretrained("checkpoint")
            self.label_list = [
                config["id2label"][i] for i in range(len(config["id2label"]))
            ]
            ##########模型初始化结束##########
        except Exception as err:  ### 此行不能删除
            ret = False  ### 此行不能删除
            err_message = "[Error] model init failed,err_message:[{}]".format(
                ExceptionMessage(err)
            )  ### 此行不能删除
            print(err_message)

        return ret, err_message

    """
    Detect         模型推理函数,注意不能自行增加删除函数入参
    text           输入单个文本,非批量
    return         字典dict
    """

    def Detect(self, text):
        """
        模型推理部分,由用户自行编写
        detect_result输出格式示例:
        {
            "predict1_code":"a01",
            "predict2_code":"b01",
            "predict3_code":"c01",
            "predict4_code":"",
            "predict5_code":""
        }
        数据格式说明:
        1.字典中key:predict1-5_code分别表示 一级标签到五级标签的标签编码
        2.字典中value均为由标签编码,类型为字符串,例如"a01",标签编码请参考比赛文档
        3.每一级标签仅有1种标签,默认缺省时为""
        """
        ### 请在try内编写推理代码,便于捕获错误
        try:
            ##########模型推理开始##########
            detect_result = {
                "predict1_code": "",
                "predict2_code": "",
                "predict3_code": "",
                "predict4_code": "",
                "predict5_code": "",
            }
            if text is None:
                print("[Error] text is None.")
            # data = {"text": text}
            # print(text)
            # for batch in data:
            result = self.predict(text)
            for i in result.keys():
                if i == 0:
                    detect_result["predict1_code"] = result[i][0]
                elif i == 1:
                    detect_result["predict2_code"] = result[i][0]
                elif i == 2:
                    detect_result["predict3_code"] = result[i][0]
                elif i == 3:
                    detect_result["predict4_code"] = result[i][0]
                elif i == 4:
                    detect_result["predict5_code"] = result[i][0]
                    # lalel_i =
                # print(result[i][0])
                # print(type(result[i]))
            # print(result)
            return detect_result

            # return detect_result
            ##########模型推理结束##########
        except Exception as err:  ### 此行不能删除
            print(
                "[Error] predictor.Detect failed.err_message:{}".format(
                    ExceptionMessage(err)
                )
            )  ### 此行不能删除
            return err  ### 此行不能删除

    # @paddle.no_grad()
    def predict(self, data):
        """
        Predicts the data labels.
        """

        tokenize_result = self.tokenizer(
            data, padding="max_length", max_length=self.max_seq_length, truncation=True
        )
        input_ids = paddle.to_tensor(tokenize_result["input_ids"])[None, :]
        token_type_ids = paddle.to_tensor(tokenize_result["token_type_ids"])[None, :]

        self.model.eval()
        with paddle.no_grad():
            logits = self.model(input_ids, token_type_ids=token_type_ids)
        pred = np.argmax(logits.numpy(), axis=-1)[0]
        label = self.label_list[pred]

        return {i: [s] for i, s in enumerate(label.split("##"))}


### 获取异常文件+行号+信息
def ExceptionMessage(err):
    err_message = (
        str(err.__traceback__.tb_frame.f_globals["__file__"])
        + ":"
        + str(err.__traceback__.tb_lineno)
        + "行:"
        + str(err)
    )
    return err_message


if __name__ == "__main__":
    ###备注说明:main函数提供给用户内测,修改后[不影响]评估
    predictor = Predictor()
    ret, err_message = predictor.InitModel()
    if ret:
        text = r'“游戏风云”频道（Gamefy）成立于2004年，是游戏类内容付费电视频道。同年12月开始试播。频道以"弘扬健康游戏文化，服务广大游戏受众"为理念， "游我所爱，任我风云"  标签'
        detect_result = predictor.Detect(text)
        print("detect_result", detect_result)
    else:
        print("[Error] InitModel failed. ret", ret, err_message)
