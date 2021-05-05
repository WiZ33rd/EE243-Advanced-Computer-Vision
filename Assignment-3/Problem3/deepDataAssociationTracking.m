clear all
v = VideoReader('atrium.mp4');
I = read(v);

save video.mat I

for i = 1:size(I,4)
    J(:,:,i) = rgb2gray(I(:,:,:,i));
end

l = 3;
offset = 130;
count = 1;
matches = [];
D1 = getSumOfDiff(J(:,:,offset-1:offset+1));
bbox1 = getDetections(D1); bbox(count).bbox = bbox1;
save bbox.mat bbox1

feat1 = getDeepFeatures(offset);
edges = [];
for i = offset:(size(J,3)-2)
    i;
    D2 = getSumOfDiff(J(:,:,i:i+2));
    bbox2 = getDetections(D2); bbox(count).bbox = bbox2;
    save bbox.mat bbox2
    v_idx = i + 1;
    feat2 = getDeepFeatures(v_idx);

    if isempty(bbox2)
        disp('NO DETECTIONS')
        continue
    end
    
    M = getDeepMatches(feat1,feat2);
    if isempty(M)
        disp('NO MATCHES')
        continue
    end
    
    showMatchedFeatures(I(:,:,:,i),I(:,:,:,i+1),[bbox1(M(:,1),1)+bbox1(M(:,1),3)/2 bbox1(M(:,1),2)+bbox1(M(:,1),4)/2], ...
        [bbox2(M(:,2),1)+bbox2(M(:,2),3)/2 bbox2(M(:,2),2)+bbox2(M(:,2),4)/2],'montage');
    for j = 1:size(M,1)
        nodes(count).bbox = bbox1(M(j,1),:);
        M(j,1) = count;
        count = count + 1;
    end
    for j = 1:size(M,1)
        edges = [edges; [M(j,1) M(j,2)+count-1]];
    end
    
%     drawnow,imshow(I(:,:,:,i)),title(num2str(i));   hold on
%     for j = 1:size(bbox1,1)
%         rectangle('Position',bbox1(j,:),'EdgeColor','r');
%     end   
    
    D1 = D2;
    bbox1 = bbox2;
    feat1 = feat2;
    %count = count + 1;
end

track_beginnings = setdiff(unique(edges(:,1)),unique(edges(:,2)));
tracklets = [];
for i = 1:length(track_beginnings)
    tracklets(i).track = track_beginnings(i);
    nextnode = edges(edges(:,1)==track_beginnings(i),2);
    while 1
        if isempty(nextnode)
            break
        end
        tracklets(i).track = [tracklets(i).track nextnode];
        nextnode = edges(edges(:,1)==nextnode,2);
    end
end

