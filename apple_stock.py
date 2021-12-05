{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.8.5"
    },
    "colab": {
      "name": "apple_stock.ipynb",
      "provenance": [],
      "include_colab_link": true
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/dldrks/autotrade/blob/main/apple_stock.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "6PKxvi90t8cr",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "9a8f16d4-932c-449a-8faa-304ba59840f8"
      },
      "source": [
        "!pip install tensorflow"
      ],
      "execution_count": 41,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: tensorflow in /usr/local/lib/python3.7/dist-packages (2.7.0)\n",
            "Requirement already satisfied: tensorboard~=2.6 in /usr/local/lib/python3.7/dist-packages (from tensorflow) (2.7.0)\n",
            "Requirement already satisfied: absl-py>=0.4.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow) (0.12.0)\n",
            "Requirement already satisfied: h5py>=2.9.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow) (3.1.0)\n",
            "Requirement already satisfied: astunparse>=1.6.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow) (1.6.3)\n",
            "Requirement already satisfied: opt-einsum>=2.3.2 in /usr/local/lib/python3.7/dist-packages (from tensorflow) (3.3.0)\n",
            "Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.21.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow) (0.22.0)\n",
            "Requirement already satisfied: wheel<1.0,>=0.32.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow) (0.37.0)\n",
            "Requirement already satisfied: typing-extensions>=3.6.6 in /usr/local/lib/python3.7/dist-packages (from tensorflow) (3.10.0.2)\n",
            "Requirement already satisfied: keras<2.8,>=2.7.0rc0 in /usr/local/lib/python3.7/dist-packages (from tensorflow) (2.7.0)\n",
            "Requirement already satisfied: grpcio<2.0,>=1.24.3 in /usr/local/lib/python3.7/dist-packages (from tensorflow) (1.42.0)\n",
            "Requirement already satisfied: libclang>=9.0.1 in /usr/local/lib/python3.7/dist-packages (from tensorflow) (12.0.0)\n",
            "Requirement already satisfied: tensorflow-estimator<2.8,~=2.7.0rc0 in /usr/local/lib/python3.7/dist-packages (from tensorflow) (2.7.0)\n",
            "Requirement already satisfied: keras-preprocessing>=1.1.1 in /usr/local/lib/python3.7/dist-packages (from tensorflow) (1.1.2)\n",
            "Requirement already satisfied: numpy>=1.14.5 in /usr/local/lib/python3.7/dist-packages (from tensorflow) (1.19.5)\n",
            "Requirement already satisfied: gast<0.5.0,>=0.2.1 in /usr/local/lib/python3.7/dist-packages (from tensorflow) (0.4.0)\n",
            "Requirement already satisfied: termcolor>=1.1.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow) (1.1.0)\n",
            "Requirement already satisfied: six>=1.12.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow) (1.15.0)\n",
            "Requirement already satisfied: wrapt>=1.11.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow) (1.13.3)\n",
            "Requirement already satisfied: flatbuffers<3.0,>=1.12 in /usr/local/lib/python3.7/dist-packages (from tensorflow) (2.0)\n",
            "Requirement already satisfied: protobuf>=3.9.2 in /usr/local/lib/python3.7/dist-packages (from tensorflow) (3.17.3)\n",
            "Requirement already satisfied: google-pasta>=0.1.1 in /usr/local/lib/python3.7/dist-packages (from tensorflow) (0.2.0)\n",
            "Requirement already satisfied: cached-property in /usr/local/lib/python3.7/dist-packages (from h5py>=2.9.0->tensorflow) (1.5.2)\n",
            "Requirement already satisfied: tensorboard-data-server<0.7.0,>=0.6.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard~=2.6->tensorflow) (0.6.1)\n",
            "Requirement already satisfied: setuptools>=41.0.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard~=2.6->tensorflow) (57.4.0)\n",
            "Requirement already satisfied: werkzeug>=0.11.15 in /usr/local/lib/python3.7/dist-packages (from tensorboard~=2.6->tensorflow) (1.0.1)\n",
            "Requirement already satisfied: tensorboard-plugin-wit>=1.6.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard~=2.6->tensorflow) (1.8.0)\n",
            "Requirement already satisfied: markdown>=2.6.8 in /usr/local/lib/python3.7/dist-packages (from tensorboard~=2.6->tensorflow) (3.3.6)\n",
            "Requirement already satisfied: google-auth<3,>=1.6.3 in /usr/local/lib/python3.7/dist-packages (from tensorboard~=2.6->tensorflow) (1.35.0)\n",
            "Requirement already satisfied: requests<3,>=2.21.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard~=2.6->tensorflow) (2.23.0)\n",
            "Requirement already satisfied: google-auth-oauthlib<0.5,>=0.4.1 in /usr/local/lib/python3.7/dist-packages (from tensorboard~=2.6->tensorflow) (0.4.6)\n",
            "Requirement already satisfied: pyasn1-modules>=0.2.1 in /usr/local/lib/python3.7/dist-packages (from google-auth<3,>=1.6.3->tensorboard~=2.6->tensorflow) (0.2.8)\n",
            "Requirement already satisfied: rsa<5,>=3.1.4 in /usr/local/lib/python3.7/dist-packages (from google-auth<3,>=1.6.3->tensorboard~=2.6->tensorflow) (4.8)\n",
            "Requirement already satisfied: cachetools<5.0,>=2.0.0 in /usr/local/lib/python3.7/dist-packages (from google-auth<3,>=1.6.3->tensorboard~=2.6->tensorflow) (4.2.4)\n",
            "Requirement already satisfied: requests-oauthlib>=0.7.0 in /usr/local/lib/python3.7/dist-packages (from google-auth-oauthlib<0.5,>=0.4.1->tensorboard~=2.6->tensorflow) (1.3.0)\n",
            "Requirement already satisfied: importlib-metadata>=4.4 in /usr/local/lib/python3.7/dist-packages (from markdown>=2.6.8->tensorboard~=2.6->tensorflow) (4.8.2)\n",
            "Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.7/dist-packages (from importlib-metadata>=4.4->markdown>=2.6.8->tensorboard~=2.6->tensorflow) (3.6.0)\n",
            "Requirement already satisfied: pyasn1<0.5.0,>=0.4.6 in /usr/local/lib/python3.7/dist-packages (from pyasn1-modules>=0.2.1->google-auth<3,>=1.6.3->tensorboard~=2.6->tensorflow) (0.4.8)\n",
            "Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests<3,>=2.21.0->tensorboard~=2.6->tensorflow) (2.10)\n",
            "Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests<3,>=2.21.0->tensorboard~=2.6->tensorflow) (3.0.4)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests<3,>=2.21.0->tensorboard~=2.6->tensorflow) (2021.10.8)\n",
            "Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.7/dist-packages (from requests<3,>=2.21.0->tensorboard~=2.6->tensorflow) (1.24.3)\n",
            "Requirement already satisfied: oauthlib>=3.0.0 in /usr/local/lib/python3.7/dist-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<0.5,>=0.4.1->tensorboard~=2.6->tensorflow) (3.1.1)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "scrolled": true,
        "id": "XSEezyipt8cy",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "3028850b-2329-47d9-bf9f-8cd3dcec2383"
      },
      "source": [
        "!pip install seaborn"
      ],
      "execution_count": 42,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: seaborn in /usr/local/lib/python3.7/dist-packages (0.11.2)\n",
            "Requirement already satisfied: pandas>=0.23 in /usr/local/lib/python3.7/dist-packages (from seaborn) (1.1.5)\n",
            "Requirement already satisfied: matplotlib>=2.2 in /usr/local/lib/python3.7/dist-packages (from seaborn) (3.2.2)\n",
            "Requirement already satisfied: numpy>=1.15 in /usr/local/lib/python3.7/dist-packages (from seaborn) (1.19.5)\n",
            "Requirement already satisfied: scipy>=1.0 in /usr/local/lib/python3.7/dist-packages (from seaborn) (1.4.1)\n",
            "Requirement already satisfied: python-dateutil>=2.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib>=2.2->seaborn) (2.8.2)\n",
            "Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.7/dist-packages (from matplotlib>=2.2->seaborn) (0.11.0)\n",
            "Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib>=2.2->seaborn) (3.0.6)\n",
            "Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib>=2.2->seaborn) (1.3.2)\n",
            "Requirement already satisfied: pytz>=2017.2 in /usr/local/lib/python3.7/dist-packages (from pandas>=0.23->seaborn) (2018.9)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.7/dist-packages (from python-dateutil>=2.1->matplotlib>=2.2->seaborn) (1.15.0)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "athSvsz5uHk7",
        "outputId": "a681cff9-15c0-413e-bc45-da84e3c2faff"
      },
      "source": [
        "!pip install yfinance"
      ],
      "execution_count": 43,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: yfinance in /usr/local/lib/python3.7/dist-packages (0.1.67)\n",
            "Requirement already satisfied: pandas>=0.24 in /usr/local/lib/python3.7/dist-packages (from yfinance) (1.1.5)\n",
            "Requirement already satisfied: multitasking>=0.0.7 in /usr/local/lib/python3.7/dist-packages (from yfinance) (0.0.10)\n",
            "Requirement already satisfied: numpy>=1.15 in /usr/local/lib/python3.7/dist-packages (from yfinance) (1.19.5)\n",
            "Requirement already satisfied: lxml>=4.5.1 in /usr/local/lib/python3.7/dist-packages (from yfinance) (4.6.4)\n",
            "Requirement already satisfied: requests>=2.20 in /usr/local/lib/python3.7/dist-packages (from yfinance) (2.23.0)\n",
            "Requirement already satisfied: python-dateutil>=2.7.3 in /usr/local/lib/python3.7/dist-packages (from pandas>=0.24->yfinance) (2.8.2)\n",
            "Requirement already satisfied: pytz>=2017.2 in /usr/local/lib/python3.7/dist-packages (from pandas>=0.24->yfinance) (2018.9)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.7/dist-packages (from python-dateutil>=2.7.3->pandas>=0.24->yfinance) (1.15.0)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests>=2.20->yfinance) (2021.10.8)\n",
            "Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests>=2.20->yfinance) (2.10)\n",
            "Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.7/dist-packages (from requests>=2.20->yfinance) (1.24.3)\n",
            "Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests>=2.20->yfinance) (3.0.4)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "0_ZStkFvt8cz"
      },
      "source": [
        "import numpy as np \n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import yfinance as yf"
      ],
      "execution_count": 44,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "20NkY_7pvkbu",
        "outputId": "be1557ee-0967-4a76-ea18-0afa4e8b3d6d"
      },
      "source": [
        "!pip intall MinMaxScalerb"
      ],
      "execution_count": 45,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "ERROR: unknown command \"intall\" - maybe you meant \"install\"\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "_uRV4yUHvyQy",
        "outputId": "128fcc7a-cc38-40d5-b28c-d38d91a26286"
      },
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/gdrive/')"
      ],
      "execution_count": 46,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Drive already mounted at /content/gdrive/; to attempt to forcibly remount, call drive.mount(\"/content/gdrive/\", force_remount=True).\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "7OHiEXjit8cz"
      },
      "source": [
        "import seaborn as sns\n",
        "from sklearn.preprocessing import MinMaxScaler"
      ],
      "execution_count": 47,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "scrolled": false,
        "id": "1TDShoiJt8c0",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "f6ba7b81-0833-4a34-858f-8c95646efe9a"
      },
      "source": [
        "apple = yf.download('AAPl')"
      ],
      "execution_count": 48,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\r[*********************100%***********************]  1 of 1 completed\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "scrolled": true,
        "id": "al0nZaSGt8c1"
      },
      "source": [
        "apple_stock = pd.DataFrame(apple)"
      ],
      "execution_count": 49,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "cfrbeU_mwfCS"
      },
      "source": [
        "apple_stock.to_csv('apple.csv')\n",
        "apple_stock = pd.read_csv('/content/apple.csv')"
      ],
      "execution_count": 50,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "aT8zajiAt8c3"
      },
      "source": [
        "apple_stock['Date'] = pd.to_datetime(apple_stock['Date'], format='%Y-%m-%d')"
      ],
      "execution_count": 51,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "V96wBI3dt8c4",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 221
        },
        "outputId": "bd439abf-7941-4805-94db-38e7945b4429"
      },
      "source": [
        "print(apple_stock.shape)\n",
        "apple_stock.tail()"
      ],
      "execution_count": 52,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "(10333, 7)\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Date</th>\n",
              "      <th>Open</th>\n",
              "      <th>High</th>\n",
              "      <th>Low</th>\n",
              "      <th>Close</th>\n",
              "      <th>Adj Close</th>\n",
              "      <th>Volume</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>10328</th>\n",
              "      <td>2021-11-29</td>\n",
              "      <td>159.369995</td>\n",
              "      <td>161.190002</td>\n",
              "      <td>158.789993</td>\n",
              "      <td>160.240005</td>\n",
              "      <td>160.240005</td>\n",
              "      <td>88748200</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>10329</th>\n",
              "      <td>2021-11-30</td>\n",
              "      <td>159.990005</td>\n",
              "      <td>165.520004</td>\n",
              "      <td>159.919998</td>\n",
              "      <td>165.300003</td>\n",
              "      <td>165.300003</td>\n",
              "      <td>174048100</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>10330</th>\n",
              "      <td>2021-12-01</td>\n",
              "      <td>167.479996</td>\n",
              "      <td>170.300003</td>\n",
              "      <td>164.529999</td>\n",
              "      <td>164.770004</td>\n",
              "      <td>164.770004</td>\n",
              "      <td>152052500</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>10331</th>\n",
              "      <td>2021-12-02</td>\n",
              "      <td>158.740005</td>\n",
              "      <td>164.199997</td>\n",
              "      <td>157.800003</td>\n",
              "      <td>163.759995</td>\n",
              "      <td>163.759995</td>\n",
              "      <td>136739200</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>10332</th>\n",
              "      <td>2021-12-03</td>\n",
              "      <td>164.020004</td>\n",
              "      <td>164.960007</td>\n",
              "      <td>159.720001</td>\n",
              "      <td>161.839996</td>\n",
              "      <td>161.839996</td>\n",
              "      <td>117938300</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "            Date        Open        High  ...       Close   Adj Close     Volume\n",
              "10328 2021-11-29  159.369995  161.190002  ...  160.240005  160.240005   88748200\n",
              "10329 2021-11-30  159.990005  165.520004  ...  165.300003  165.300003  174048100\n",
              "10330 2021-12-01  167.479996  170.300003  ...  164.770004  164.770004  152052500\n",
              "10331 2021-12-02  158.740005  164.199997  ...  163.759995  163.759995  136739200\n",
              "10332 2021-12-03  164.020004  164.960007  ...  161.839996  161.839996  117938300\n",
              "\n",
              "[5 rows x 7 columns]"
            ]
          },
          "metadata": {},
          "execution_count": 52
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "vEnrgeW5t8c6",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 507
        },
        "outputId": "2f1c069a-3e91-473b-b5c8-afc482af8157"
      },
      "source": [
        "plt.figure(figsize=(16, 9))\n",
        "sns.lineplot(y=apple_stock['Close'], x=apple_stock['Date'])"
      ],
      "execution_count": 54,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.axes._subplots.AxesSubplot at 0x7fe0cadca950>"
            ]
          },
          "metadata": {},
          "execution_count": 54
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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3fCz+fmJvyRp0pkZSSMhs9qzzGqf+NYKEWluZnbYX5crIiIijaHRTocvBU4wswVmlgYuB+6oc5lERERERESkwTRUy6xzrmJm1wG/J/5rnh8651bWuVgiIiIiIiLSYBoqmQVwzv0W+G29yyEiIiIiIiKNq9G6GYuIiIiIiIjsl5JZERERERERaTpKZkVERERERKTpKJkVERERERGRpqNkVkRERERERJqOklkRERERERFpOkpmRUREREREpOkomRUREREREZGmo2RWREREREREmo6SWREREREREWk6SmZFRERERESk6SiZFRERERERkaajZFZERERERESajpJZERERERERaTpKZkVERERERKTpKJkVERERERGRpqNkVkRERERERJqOOefqXYZxM7OtwPpDsOpOYNshWK+MnWLROBSLxqFYNA7FonEoFo1DsWgcikXjUCzG71jn3IzRJjR1MnuomNky59ySepdDFItGolg0DsWicSgWjUOxaByKReNQLBqHYnFoqJuxiIiIiIiINB0lsyIiIiIiItJ0lMyO7vv1LoDUKBaNQ7FoHIpF41AsGodi0TgUi8ahWDQOxeIQ0DWzIiIiIiIi0nTUMisiIiIiIiJN54hJZs3sh2a2xcxWDBv3BjP7o5k9a2a/NrP2ZHzKzG5Lxq82s78atsw7zOx5M1tjZtfXY1ua3QTGYl0y/mkzW1aPbWl2Y4xF2sxuTcY/Y2bnD1vmjGT8GjP7pplZHTanqU1gLO5LvqOeToaZddicpmVmc83sXjNbZWYrzeyTyfhpZvYHM3sxeZyajLdkn19jZsvN7PRh67o6mf9FM7u6XtvUzCY4HuGw4+KOem1TsxpHLBYl319FM/vsbutSXeogTHAsVJc6COOIxRXJd9OzZvaImb1h2Lp0XIyHc+6IGIC3AKcDK4aNWwqclzz/CHBj8vyDwO3J8xywDpgP+MBaYCGQBp4BFtd725ptmIhYJK/XAZ313p5mHsYYi08AtybPZwJPAF7y+nHgbMCA3wEX13vbmm2YwFjcByyp9/Y06wDMAU5PnrcBLwCLga8D1yfjrwf+Nnn+zmSft+QYeCwZPw14KXmcmjyfWu/ta7ZhouKRTOuv9/Y08zCOWMwE3gR8DfjssPWoLtUgsUimrUN1qcmMxTnV3wLg4mG/GTouxjkcMS2zzrkHgB27jT4ReCB5/gfg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\n",
            "text/plain": [
              "<Figure size 1152x648 with 1 Axes>"
            ]
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "kRh4iDM-t8c6",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 523
        },
        "outputId": "7b360d5e-bdc1-41d1-ad68-8a9b6d4141b9"
      },
      "source": [
        "plt.figure(figsize=(16, 9))\n",
        "sns.lineplot(y=apple_stock['Volume'], x=apple_stock['Date'])"
      ],
      "execution_count": 55,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.axes._subplots.AxesSubplot at 0x7fe0cc1e8190>"
            ]
          },
          "metadata": {},
          "execution_count": 55
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 1152x648 with 1 Axes>"
            ]
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "-oZdldSLt8c7"
      },
      "source": [
        "scaler = MinMaxScaler()\n",
        "scale_cols = ['Open', 'High', 'Low', 'Close', 'Volume']\n",
        "apple_stock_scaled = scaler.fit_transform(apple_stock[scale_cols])"
      ],
      "execution_count": 57,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "lPhYu04Ot8c8"
      },
      "source": [
        "apple_stock_scaled = pd.DataFrame(apple_stock_scaled)\n",
        "apple_stock_scaled_columns = scale_cols"
      ],
      "execution_count": 58,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "KrJCO6Zst8c8"
      },
      "source": [
        "TEST_SIZE = 200\n",
        "WINDOW_SIZE = 20\n",
        "\n",
        "\n",
        "train = apple_stock_scaled[:-TEST_SIZE]\n",
        "test = apple_stock_scaled[-TEST_SIZE:]"
      ],
      "execution_count": 61,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Cq8mu7OKt8c9"
      },
      "source": [
        "def make_dataset(data, label, window_size=20):\n",
        "    feature_list = []\n",
        "    label_list = []\n",
        "    for i in range(len(data) - window_size):\n",
        "        feature_list.append(np.array(data.iloc[i:i+window_size]))\n",
        "        label_list.append(np.array(label.iloc[i+window_size]))\n",
        "    return np.array(feature_list), np.array(label_list)"
      ],
      "execution_count": 62,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ILqs6xEot8c9"
      },
      "source": [
        "train.columns = ['Open', 'High', 'Low', 'Close', 'Volume']"
      ],
      "execution_count": 63,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "szqvQNEit8c-",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "15a8e105-7151-40b3-b17c-d3a80214c508"
      },
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "\n",
        "feature_cols = ['Open', 'High', 'Low',  'Volume']\n",
        "label_cols = ['Close']\n",
        "\n",
        "train_feature = train[feature_cols]\n",
        "train_label = train[label_cols]\n",
        "\n",
        "train_feature, train_label = make_dataset(train_feature, train_label, 20)\n",
        "\n",
        "x_train, x_valid, y_train, y_valid = train_test_split(train_feature, train_label, test_size=0.2)\n",
        "x_train.shape, x_valid.shape"
      ],
      "execution_count": 64,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "((8090, 20, 4), (2023, 20, 4))"
            ]
          },
          "metadata": {},
          "execution_count": 64
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "8qFx4Zf3t8c-"
      },
      "source": [
        "test.columns = ['Open', 'High', 'Low', 'Close', 'Volume']"
      ],
      "execution_count": 65,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "8pkrKj-Nt8c-",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "6408d4a7-45cb-44f6-f99e-f56ee5e84ac8"
      },
      "source": [
        "test_feature = test[feature_cols]\n",
        "test_label = test[label_cols]\n",
        "\n",
        "test_feature.shape, test_label.shape"
      ],
      "execution_count": 66,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "((200, 4), (200, 1))"
            ]
          },
          "metadata": {},
          "execution_count": 66
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "scrolled": true,
        "id": "YiFz3Fk1t8c_",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "10a87d1d-1ad1-49cb-fe4a-d68b6e75b421"
      },
      "source": [
        "test_feature, test_label = make_dataset(test_feature, test_label, 20)\n",
        "test_feature.shape, test_label.shape"
      ],
      "execution_count": 67,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "((180, 20, 4), (180, 1))"
            ]
          },
          "metadata": {},
          "execution_count": 67
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "mpKhfQyqt8c_"
      },
      "source": [
        "from keras.models import Sequential\n",
        "from keras.layers import Dense\n",
        "from keras.callbacks import EarlyStopping, ModelCheckpoint\n",
        "from keras.layers import LSTM\n",
        "\n",
        "model = Sequential()\n",
        "model.add(LSTM(16, \n",
        "               input_shape=(train_feature.shape[1], train_feature.shape[2]), \n",
        "               activation='relu', \n",
        "               return_sequences=False)\n",
        "          )\n",
        "\n",
        "model.add(Dense(1))"
      ],
      "execution_count": 68,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "2MmmCjfst8dA",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "e4d7745d-b446-4e86-95ef-896f9cf1e8cb"
      },
      "source": [
        "import os\n",
        "\n",
        "model.compile(loss='mean_squared_error', optimizer='adam')\n",
        "early_stop = EarlyStopping(monitor='val_loss', patience=5)\n",
        "\n",
        "model_path = 'model'\n",
        "filename = os.path.join(model_path, 'tmp_checkpoint.h5')\n",
        "checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')\n",
        "\n",
        "history = model.fit(x_train, y_train, \n",
        "                                    epochs=200, \n",
        "                                    batch_size=16,\n",
        "                                    validation_data=(x_valid, y_valid), \n",
        "                                    callbacks=[early_stop, checkpoint])"
      ],
      "execution_count": 69,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/200\n",
            "506/506 [==============================] - ETA: 0s - loss: 7.0570e-04\n",
            "Epoch 00001: val_loss improved from inf to 0.00003, saving model to model/tmp_checkpoint.h5\n",
            "506/506 [==============================] - 6s 8ms/step - loss: 7.0570e-04 - val_loss: 2.8676e-05\n",
            "Epoch 2/200\n",
            "506/506 [==============================] - ETA: 0s - loss: 3.2780e-05\n",
            "Epoch 00002: val_loss did not improve from 0.00003\n",
            "506/506 [==============================] - 4s 8ms/step - loss: 3.2780e-05 - val_loss: 3.0518e-05\n",
            "Epoch 3/200\n",
            "504/506 [============================>.] - ETA: 0s - loss: 2.6627e-05\n",
            "Epoch 00003: val_loss did not improve from 0.00003\n",
            "506/506 [==============================] - 4s 8ms/step - loss: 2.6548e-05 - val_loss: 4.1189e-05\n",
            "Epoch 4/200\n",
            "503/506 [============================>.] - ETA: 0s - loss: 2.8362e-05\n",
            "Epoch 00004: val_loss improved from 0.00003 to 0.00002, saving model to model/tmp_checkpoint.h5\n",
            "506/506 [==============================] - 4s 8ms/step - loss: 2.8227e-05 - val_loss: 2.1330e-05\n",
            "Epoch 5/200\n",
            "501/506 [============================>.] - ETA: 0s - loss: 2.7888e-05\n",
            "Epoch 00005: val_loss did not improve from 0.00002\n",
            "506/506 [==============================] - 4s 8ms/step - loss: 2.7883e-05 - val_loss: 2.5195e-05\n",
            "Epoch 6/200\n",
            "503/506 [============================>.] - ETA: 0s - loss: 2.5590e-05\n",
            "Epoch 00006: val_loss did not improve from 0.00002\n",
            "506/506 [==============================] - 4s 8ms/step - loss: 2.6183e-05 - val_loss: 3.6244e-05\n",
            "Epoch 7/200\n",
            "501/506 [============================>.] - ETA: 0s - loss: 2.6123e-05\n",
            "Epoch 00007: val_loss did not improve from 0.00002\n",
            "506/506 [==============================] - 4s 8ms/step - loss: 2.6113e-05 - val_loss: 2.2524e-05\n",
            "Epoch 8/200\n",
            "501/506 [============================>.] - ETA: 0s - loss: 2.4879e-05\n",
            "Epoch 00008: val_loss improved from 0.00002 to 0.00002, saving model to model/tmp_checkpoint.h5\n",
            "506/506 [==============================] - 4s 8ms/step - loss: 2.4773e-05 - val_loss: 2.1210e-05\n",
            "Epoch 9/200\n",
            "500/506 [============================>.] - ETA: 0s - loss: 2.3281e-05\n",
            "Epoch 00009: val_loss improved from 0.00002 to 0.00002, saving model to model/tmp_checkpoint.h5\n",
            "506/506 [==============================] - 4s 8ms/step - loss: 2.3099e-05 - val_loss: 1.9005e-05\n",
            "Epoch 10/200\n",
            "506/506 [==============================] - ETA: 0s - loss: 2.1480e-05\n",
            "Epoch 00010: val_loss improved from 0.00002 to 0.00002, saving model to model/tmp_checkpoint.h5\n",
            "506/506 [==============================] - 4s 8ms/step - loss: 2.1480e-05 - val_loss: 1.7143e-05\n",
            "Epoch 11/200\n",
            "503/506 [============================>.] - ETA: 0s - loss: 2.3606e-05\n",
            "Epoch 00011: val_loss did not improve from 0.00002\n",
            "506/506 [==============================] - 4s 8ms/step - loss: 2.3611e-05 - val_loss: 2.5171e-05\n",
            "Epoch 12/200\n",
            "506/506 [==============================] - ETA: 0s - loss: 2.0604e-05\n",
            "Epoch 00012: val_loss did not improve from 0.00002\n",
            "506/506 [==============================] - 4s 8ms/step - loss: 2.0604e-05 - val_loss: 2.3034e-05\n",
            "Epoch 13/200\n",
            "503/506 [============================>.] - ETA: 0s - loss: 2.1519e-05\n",
            "Epoch 00013: val_loss improved from 0.00002 to 0.00001, saving model to model/tmp_checkpoint.h5\n",
            "506/506 [==============================] - 4s 8ms/step - loss: 2.1473e-05 - val_loss: 1.3074e-05\n",
            "Epoch 14/200\n",
            "502/506 [============================>.] - ETA: 0s - loss: 1.9681e-05\n",
            "Epoch 00014: val_loss did not improve from 0.00001\n",
            "506/506 [==============================] - 4s 8ms/step - loss: 2.0217e-05 - val_loss: 2.3021e-05\n",
            "Epoch 15/200\n",
            "502/506 [============================>.] - ETA: 0s - loss: 1.9934e-05\n",
            "Epoch 00015: val_loss did not improve from 0.00001\n",
            "506/506 [==============================] - 4s 8ms/step - loss: 1.9839e-05 - val_loss: 1.6197e-05\n",
            "Epoch 16/200\n",
            "502/506 [============================>.] - ETA: 0s - loss: 2.0726e-05\n",
            "Epoch 00016: val_loss did not improve from 0.00001\n",
            "506/506 [==============================] - 4s 8ms/step - loss: 2.0935e-05 - val_loss: 2.1220e-05\n",
            "Epoch 17/200\n",
            "499/506 [============================>.] - ETA: 0s - loss: 1.8934e-05\n",
            "Epoch 00017: val_loss improved from 0.00001 to 0.00001, saving model to model/tmp_checkpoint.h5\n",
            "506/506 [==============================] - 4s 8ms/step - loss: 1.8796e-05 - val_loss: 1.2232e-05\n",
            "Epoch 18/200\n",
            "504/506 [============================>.] - ETA: 0s - loss: 1.9520e-05\n",
            "Epoch 00018: val_loss did not improve from 0.00001\n",
            "506/506 [==============================] - 4s 8ms/step - loss: 1.9511e-05 - val_loss: 1.4416e-05\n",
            "Epoch 19/200\n",
            "501/506 [============================>.] - ETA: 0s - loss: 2.0455e-05\n",
            "Epoch 00019: val_loss did not improve from 0.00001\n",
            "506/506 [==============================] - 4s 8ms/step - loss: 2.0327e-05 - val_loss: 1.4734e-05\n",
            "Epoch 20/200\n",
            "501/506 [============================>.] - ETA: 0s - loss: 1.5820e-05\n",
            "Epoch 00020: val_loss did not improve from 0.00001\n",
            "506/506 [==============================] - 4s 8ms/step - loss: 1.6201e-05 - val_loss: 2.5686e-05\n",
            "Epoch 21/200\n",
            "506/506 [==============================] - ETA: 0s - loss: 1.7264e-05\n",
            "Epoch 00021: val_loss did not improve from 0.00001\n",
            "506/506 [==============================] - 4s 8ms/step - loss: 1.7264e-05 - val_loss: 1.3017e-05\n",
            "Epoch 22/200\n",
            "499/506 [============================>.] - ETA: 0s - loss: 1.6369e-05\n",
            "Epoch 00022: val_loss did not improve from 0.00001\n",
            "506/506 [==============================] - 4s 8ms/step - loss: 1.6705e-05 - val_loss: 2.2305e-05\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "z6Gv08JFt8dB",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "530fa70b-1a7f-49c4-f246-cb0646250976"
      },
      "source": [
        "model.load_weights(filename)\n",
        "pred = model.predict(test_feature)\n",
        "\n",
        "pred.shape"
      ],
      "execution_count": 70,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(180, 1)"
            ]
          },
          "metadata": {},
          "execution_count": 70
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "EeINNk9vt8dB",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 537
        },
        "outputId": "ae7f9767-b38e-4e26-807d-db5d7221bf54"
      },
      "source": [
        "plt.figure(figsize=(12, 9))\n",
        "plt.plot(test_label, label = 'actual')\n",
        "plt.plot(pred, label = 'prediction')\n",
        "plt.legend()\n",
        "plt.show()"
      ],
      "execution_count": 71,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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OW+55RauueYa5s2bx5133jnGf1siIiIylbV2WEFyUVYyd715ES+daOLPuyNW3U5YCeO9gGH7511Qu39kn3P6Mrg2eslDyNGjR3nggQdYv34973//+8MZ3tzcXHbv3k1jYyO33HILzz77LKmpqXz729/mnnvu4c477+SOO+7g+eefZ+7cueHx1QN98pOfZMOGDTz22GP4/X7a29v51re+xYEDB9i7dy9gBesh9913H4ZhsH//fo4cOcLVV1/NsWPHANi7dy979uwhMTGRBQsW8IlPfIKSkpJILysiIiIyolo6uslMdgDwrrUzeeSVCn77SgW3XlA8ziuL34TobjFZlJSUsH79egBuv/12XnzxRYBw0PvKK69w6NAh1q9fz8qVK/n1r39NRUUFR44coaysjHnz5mEYBrfffnvE53/++ef5yEc+AoDdbiczMzPmel588cXwcy1cuJBZs2aFg+SNGzeSmZlJUlISixcvpqJichbNi4iIyOTj7vCRleIEwGYzuHFlEXsrWyZVbfLkyyTHkfEdLYZhRPw+NTUVANM0ueqqq/j973/f77pQFngsJSYmhr+22+309PSM+RpERERkamrx+phfkB7+/rrlM/j2U0f4274aPnr53HFcWfyUSR6GM2fOsG3bNgB+97vfcckll/Q7v27dOl566SVOnDgBgMfj4dixYyxcuJDy8nJOnjwJMCiIDtm4cSM//elPAWsToNvtJj09nba2tojXX3rppTzyyCMAHDt2jDNnzrBgwYLX/0ZFREREXge310dGsNwCoCQnhVUzs/jba2fHcVXDoyB5GBYsWMB9993HokWLaG5uDpdGhOTn5/PQQw9x2223sXz5ci6++GKOHDlCUlIS999/P295y1tYvXo1BQUFEZ//3nvvZdOmTSxbtowLLriAQ4cOkZuby/r161m6dCmf+9zn+l3/0Y9+lEAgwLJly3j729/OQw891C+DLCIiIjLWevwB2rp6yEpx9Dt+/fJCDp9t5UT95JjCZ0y0dhxr1qwxd+7c2e/Y4cOHWbRo0TityFJeXs51113HgQMHxnUdI20i/N2KiIjI+cPl6Wb115/h7usX8771ZeHjda2drPvmc/y/jfP41JXzx3GFvQzD2GWa5ppI55RJFhEREZER4w5O2xuYSZ6WkcRFZTn87bWaSdEzWUFynEpLS8+7LLKIiIjISGvx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\n",
            "text/plain": [
              "<Figure size 864x648 with 1 Axes>"
            ]
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "eXFnsC2tt8dB"
      },
      "source": [
        ""
      ],
      "execution_count": null,
      "outputs": []
    }
  ]
}