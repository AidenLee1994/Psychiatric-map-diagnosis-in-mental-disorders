clf reset;
f0 = gcf;
clc;


%common 
map_size=[20 10];

epoch = 300;

percent =0.2;


%make file for save pic
pic_file_path=strcat('../',num2str(percent),'_pic/');
%pic_file_path=strcat('../',num2str(percent),'_pic_no/');
mkdir(pic_file_path)


%inpute data
try
    file_name=char(strcat('../',num2str(percent),'/','snp_sample.data'));
catch
    disp("file not found!");
end 

sD = som_read_data(file_name);  

sMap=som_randinit(sD,'msize',map_size);

sMap  = som_seqtrain(sMap,sD,'trainlen',epoch);

column=sMap.comp_names;

columns=reshape(column,1,length(column));

data=num2cell(sMap.codebook);

Data=table(cat(1,columns,data));

writetable(Data,char(strcat('../',num2str(percent),'/',num2str(percent),'_prototype.csv')));




colormap(jet(64));

for i = 1:length(sD.comp_names)

    som_show(sMap,'footnote','','comp',i);
    %som_show(sMap,'footnote','');
    
    saveas(gcf,strcat(pic_file_path,sD.comp_names{i},'.png'));
    
end

