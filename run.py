import disneylandClient
import os

client = disneylandClient.new_client()
ret = client.ListJobs(disneylandClient.ListJobsRequest(how_many=10, kind="ock"))
print(ret)